//! Safe wrapper around an opened Vimba X camera.
//!
//! [`Camera`] is generic over any [`VmbRuntime`] and drives the
//! announce / queue / capture-start dance entirely through the port —
//! with zero FFI or `unsafe` in this module.

use std::path::Path;
use std::sync::Arc;

use crate::callback::FrameCallback;
use crate::frame::Frame;
use crate::port::VmbRuntime;
use crate::types::{CameraHandle, FrameCallbackId, FrameSlotId};
use crate::{Result, VmbError};

/// Command features issued during the capture lifecycle. These GenICam
/// names are stable across Vimba SDK versions; hoisting them to consts
/// makes SDK-upgrade audits a one-grep review.
const FEATURE_ACQUISITION_START: &str = "AcquisitionStart";
const FEATURE_ACQUISITION_STOP: &str = "AcquisitionStop";

/// An in-progress capture session — the set of announced slots plus the
/// installed callback identifier, both of which must be unwound on
/// teardown.
struct CaptureSession {
    callback_id: FrameCallbackId,
    #[allow(dead_code)]
    slots: Vec<FrameSlotId>,
}

/// Open handle to a Vimba camera.
///
/// Dropping the camera cleanly ends any running capture (via
/// [`Camera::stop_capture`]) and closes the adapter-side resources.
pub struct Camera<R: VmbRuntime> {
    runtime: Arc<R>,
    handle: CameraHandle,
    id: String,
    session: Option<CaptureSession>,
}

impl<R: VmbRuntime> Camera<R> {
    /// Open a camera by its transport-layer ID. Usually called via
    /// [`VmbSystem::open_camera`](crate::system::VmbSystem::open_camera).
    pub fn open(runtime: Arc<R>, id: &str) -> Result<Self> {
        let handle = runtime.open_camera(id)?;
        Ok(Self {
            runtime,
            handle,
            id: id.to_string(),
            session: None,
        })
    }

    /// The camera ID originally passed to [`Camera::open`].
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Query Vimba camera int feature.
    pub fn get_feature_int(&self, name: &str) -> Result<i64> {
        self.runtime.get_feature_int(self.handle, name)
    }

    /// Query Vimba camera float feature.
    pub fn get_feature_float(&self, name: &str) -> Result<f64> {
        self.runtime.get_feature_float(self.handle, name)
    }

    /// Set Vimba camera float feature.
    pub fn set_feature_float(&self, name: &str, value: f64) -> Result<()> {
        self.runtime.set_feature_float(self.handle, name, value)
    }

    /// Set Vimba camera enum feature.
    pub fn set_feature_enum(&self, name: &str, value: &str) -> Result<()> {
        self.runtime.set_feature_enum(self.handle, name, value)
    }

    /// Load a Vimba settings XML (day/night profile).
    pub fn load_settings(&self, path: &Path) -> Result<()> {
        self.runtime.load_settings(self.handle, path)
    }

    /// Start continuous capture.
    ///
    /// The closure is invoked for every received frame; it MUST be fast
    /// and immediately copy the frame bytes (the adapter re-queues the
    /// buffer as soon as the callback returns). The closure may run on
    /// any thread the adapter chooses and must be `Send + Sync`.
    ///
    /// `num_buffers` is the number of frame buffers to pre-announce; 4
    /// is a reasonable default.
    ///
    /// # Cleanup contract
    ///
    /// All resources claimed between the first `announce_frame` and the
    /// final `Ok(())` are unwound on any error path before returning.
    /// This guarantees `self.session` is only populated when the
    /// adapter is fully primed — preventing a latent use-after-free
    /// where the SDK could otherwise hold pointers into callback
    /// allocations that get dropped when `Camera::drop` skips
    /// `stop_capture`.
    pub fn start_capture<F>(&mut self, num_buffers: usize, callback: F) -> Result<()>
    where
        F: for<'a> Fn(&Frame<'a>) + Send + Sync + 'static,
    {
        if self.session.is_some() {
            return Err(VmbError::CaptureAlreadyRunning);
        }

        let payload = self.runtime.payload_size(self.handle)?;
        let callback = Arc::new(FrameCallback::new(callback));
        let callback_id = self.runtime.install_frame_callback(callback);

        let mut slots: Vec<FrameSlotId> = Vec::with_capacity(num_buffers);

        let result: Result<()> = (|| {
            for _ in 0..num_buffers {
                let slot = self.runtime.announce_frame(self.handle, payload)?;
                slots.push(slot);
            }
            self.runtime.capture_start(self.handle)?;
            for slot in &slots {
                self.runtime.queue_frame(self.handle, *slot, callback_id)?;
            }
            self.runtime
                .run_feature_command(self.handle, FEATURE_ACQUISITION_START)?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.session = Some(CaptureSession { callback_id, slots });
                Ok(())
            }
            Err(e) => {
                // Best-effort teardown — the domain always calls all
                // three even if one wasn't reached, since each is
                // documented as a safe no-op otherwise.
                self.runtime.capture_end(self.handle);
                self.runtime.capture_queue_flush(self.handle);
                self.runtime.frame_revoke_all(self.handle);
                self.runtime.uninstall_frame_callback(callback_id);
                Err(e)
            }
        }
    }

    /// Stop an in-progress capture. Safe to call when no capture is
    /// running — the call is a no-op in that case.
    pub fn stop_capture(&mut self) -> Result<()> {
        let Some(session) = self.session.take() else {
            return Ok(());
        };
        // Best-effort teardown. Errors on these calls are deliberately
        // swallowed because we cannot recover from a partial teardown
        // failure mid-shutdown.
        let _ = self
            .runtime
            .run_feature_command(self.handle, FEATURE_ACQUISITION_STOP);
        self.runtime.capture_end(self.handle);
        self.runtime.capture_queue_flush(self.handle);
        self.runtime.frame_revoke_all(self.handle);
        self.runtime.uninstall_frame_callback(session.callback_id);
        Ok(())
    }
}

impl<R: VmbRuntime> Drop for Camera<R> {
    fn drop(&mut self) {
        if self.session.is_some() {
            let _ = self.stop_capture();
        }
        self.runtime.close_camera(self.handle);
    }
}

impl<R: VmbRuntime> std::fmt::Debug for Camera<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Camera")
            .field("id", &self.id)
            .field("handle", &self.handle)
            .field("capture_running", &self.session.is_some())
            .finish()
    }
}
