//! The [`VmbRuntime`] port and the callback wrappers the domain hands to
//! adapters.
//!
//! `VmbRuntime` is the only trait the domain calls through. Two concrete
//! implementations exist in sibling crates:
//!
//! * `vmb-ffi::VmbFfiRuntime` — production adapter that links against
//!   `libVmbC`. All `unsafe` FFI calls live there.
//! * `vmb-fake::FakeVmbRuntime` — in-memory adapter used for unit tests.
//!
//! ## Ownership of user callbacks
//!
//! User closures (frame callbacks, discovery callbacks) are handed to the
//! runtime via the `install_*_callback` methods which return opaque
//! [`FrameCallbackId`] / [`DiscoveryCallbackId`] values. The domain
//! stores the IDs and later hands them back in operations such as
//! [`VmbRuntime::queue_frame`]. When the domain tears a capture down, it
//! calls the matching `uninstall_*_callback` so the runtime can release
//! the closure.
//!
//! This keeps closure storage — the place where FFI adapters traditionally
//! juggle raw `*mut c_void` pointers — entirely behind the port.
//!
//! ## Call ordering invariants
//!
//! * `startup` must be called before any other method; a failed `startup`
//!   leaves the runtime in its "not started" state and may be retried.
//! * `shutdown` is infallible and must be safe to call from `Drop`. It
//!   must tolerate being invoked without a prior successful `startup`.
//! * `close_camera` / `capture_end` / `capture_queue_flush` /
//!   `frame_revoke_all` / `unregister_discovery` are "best-effort"
//!   teardown calls: they must not fail and must tolerate being invoked
//!   in an out-of-order state (e.g. after a partial startup of capture).
//! * `queue_frame` must only be called between `capture_start` and
//!   `capture_end`. The adapter is free to enforce this at runtime (and
//!   the fake does so aggressively).

use std::path::Path;
use std::sync::Arc;

use crate::callback::FrameCallback;
use crate::types::{
    CameraHandle, CameraInfo, DiscoveryCallbackId, DiscoveryEvent, DiscoveryRegistrationHandle,
    FrameCallbackId, FrameSlotId,
};
use crate::Result;

/// Type-erased discovery callback stored in a [`VmbRuntime`] registry.
///
/// The closure is invoked from whatever thread the runtime fires the
/// callback on (an SDK worker thread in `vmb-ffi`, the caller's thread in
/// `vmb-fake`), so it must be `Send + Sync + 'static`.
pub struct DiscoveryCallback {
    inner: Box<dyn Fn(DiscoveryEvent) + Send + Sync + 'static>,
}

impl DiscoveryCallback {
    /// Wrap a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(DiscoveryEvent) + Send + Sync + 'static,
    {
        Self { inner: Box::new(f) }
    }

    /// Invoke the closure with the given event. Intended for adapter use.
    pub fn invoke(&self, event: DiscoveryEvent) {
        (self.inner)(event);
    }
}

/// The port through which the domain layer talks to every backend.
pub trait VmbRuntime: Send + Sync + 'static {
    // -- Lifecycle -----------------------------------------------------

    /// Start the runtime. May only succeed once at a time; a second call
    /// before the matching [`VmbRuntime::shutdown`] must return
    /// [`crate::VmbError::AlreadyStarted`].
    fn startup(&self) -> Result<()>;

    /// Stop the runtime. Infallible; safe to call from `Drop`.
    fn shutdown(&self);

    // -- Cameras -------------------------------------------------------

    /// Enumerate currently-visible cameras.
    fn list_cameras(&self) -> Result<Vec<CameraInfo>>;

    /// Open a camera by its transport-layer ID.
    fn open_camera(&self, id: &str) -> Result<CameraHandle>;

    /// Close an opened camera. Best-effort; must not fail.
    fn close_camera(&self, h: CameraHandle);

    /// Load a Vimba settings XML onto an opened camera.
    fn load_settings(&self, h: CameraHandle, path: &Path) -> Result<()>;

    /// Request a feature from an opened camera.
    fn get_feature_float(&self, h: CameraHandle, name: &str) -> Result<f64>;
    
    /// Set a float feature on an opened camera.
    fn set_feature_float(&self, h: CameraHandle, name: &str, value: f64) -> Result<()>;

    /// Set an enum feature on an opened camera.
    fn set_feature_enum(&self, h: CameraHandle, name: &str, value: &str) -> Result<()>;

    /// Run a GenICam feature command (e.g. `"AcquisitionStart"`).
    fn run_feature_command(&self, h: CameraHandle, name: &str) -> Result<()>;

    // -- Capture -------------------------------------------------------

    /// Query the required payload (frame buffer) size for an opened
    /// camera.
    fn payload_size(&self, h: CameraHandle) -> Result<u32>;

    /// Announce a frame slot of `size` bytes. Returns an opaque
    /// [`FrameSlotId`] the domain will feed back into [`queue_frame`].
    fn announce_frame(&self, h: CameraHandle, size: u32) -> Result<FrameSlotId>;

    /// Begin a capture session. Must be paired with [`capture_end`].
    fn capture_start(&self, h: CameraHandle) -> Result<()>;

    /// Hand an announced frame slot to the runtime and associate it
    /// with a previously-installed [`FrameCallback`]. The runtime
    /// invokes the callback for every completed frame received into the
    /// slot.
    fn queue_frame(&self, h: CameraHandle, slot: FrameSlotId, cb: FrameCallbackId) -> Result<()>;

    /// End a capture session. Best-effort; must not fail.
    fn capture_end(&self, h: CameraHandle);

    /// Flush the per-camera frame queue. Best-effort; must not fail.
    fn capture_queue_flush(&self, h: CameraHandle);

    /// Revoke every announced frame slot for a camera. Best-effort;
    /// must not fail.
    fn frame_revoke_all(&self, h: CameraHandle);

    // -- Discovery -----------------------------------------------------

    /// Register a discovery subscription for the previously-installed
    /// callback identified by `cb`. The runtime is free to fire the
    /// callback on any thread.
    fn register_discovery(&self, cb: DiscoveryCallbackId) -> Result<DiscoveryRegistrationHandle>;

    /// Tear down a discovery subscription. Best-effort; must not fail.
    fn unregister_discovery(&self, r: DiscoveryRegistrationHandle);

    // -- Callback installation ----------------------------------------
    //
    // The runtime owns an internal registry of user closures so that
    // adapters can hand raw C trampolines an opaque `u64` identifier
    // instead of a fat trait-object pointer. The domain calls
    // `install_*` before handing the returned ID to `queue_frame` /
    // `register_discovery`, and calls the matching `uninstall_*` when
    // it no longer needs the callback.

    /// Install a user frame callback and return its ID.
    fn install_frame_callback(&self, cb: Arc<FrameCallback>) -> FrameCallbackId;

    /// Uninstall a previously-installed frame callback. Best-effort;
    /// must not fail.
    fn uninstall_frame_callback(&self, id: FrameCallbackId);

    /// Install a user discovery callback and return its ID.
    fn install_discovery_callback(&self, cb: Arc<DiscoveryCallback>) -> DiscoveryCallbackId;

    /// Uninstall a previously-installed discovery callback.
    fn uninstall_discovery_callback(&self, id: DiscoveryCallbackId);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_callback_dispatches() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let cb = DiscoveryCallback::new(move |event| {
            if matches!(event, DiscoveryEvent::Detected(_)) {
                c.fetch_add(1, Ordering::SeqCst);
            }
        });
        cb.invoke(DiscoveryEvent::Detected("cam-a".into()));
        cb.invoke(DiscoveryEvent::Missing("cam-b".into()));
        cb.invoke(DiscoveryEvent::Detected("cam-c".into()));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
