//! Programmable in-memory [`VmbRuntime`] for unit tests.
//!
//! [`FakeVmbRuntime`] is the test double for code that consumes
//! [`vmb_core::VmbRuntime`]. It records every port call, lets tests
//! drive discovery / frame callbacks synchronously, and supports
//! injecting per-method failures.
//!
//! ```no_run
//! use std::sync::Arc;
//! use vmb_core::{VmbSystem, DiscoveryEvent};
//! use vmb_fake::FakeVmbRuntime;
//!
//! let fake = FakeVmbRuntime::new();
//! let system = VmbSystem::startup(fake.clone()).unwrap();
//! let _reg = system.register_discovery(|event| {
//!     if let DiscoveryEvent::Detected(id) = event {
//!         println!("new camera: {id}");
//!     }
//! }).unwrap();
//! fake.emit_discovery(DiscoveryEvent::Detected("cam-a".into()));
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vmb_core::{
    error::VmbError, port::VmbRuntime, CameraHandle, CameraInfo, DiscoveryCallback,
    DiscoveryCallbackId, DiscoveryEvent, DiscoveryRegistrationHandle, Frame, FrameCallback,
    FrameCallbackId, FrameSlotId, PixelFormat, Result,
};

/// One recorded port-method invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeCall {
    /// `startup` was called.
    Startup,
    /// `shutdown` was called.
    Shutdown,
    /// `list_cameras` was called.
    ListCameras,
    /// `open_camera(id)` was called.
    OpenCamera(String),
    /// `close_camera(h)` was called.
    CloseCamera(CameraHandle),
    /// `load_settings(h, path)` was called.
    LoadSettings(CameraHandle, PathBuf),
    /// `get_feature_float(h, name)` was called.
    GetFeatureFloat(CameraHandle, String),
    /// `set_feature_float(h, name, value)` was called.
    SetFeatureFloat(CameraHandle, String, u64),
    /// `set_feature_enum(h, name, value)` was called.
    SetFeatureEnum(CameraHandle, String, String),
    /// `run_feature_command(h, name)` was called.
    RunFeatureCommand(CameraHandle, String),
    /// `payload_size(h)` was called.
    PayloadSize(CameraHandle),
    /// `announce_frame(h, size)` was called.
    AnnounceFrame(CameraHandle, u32),
    /// `capture_start(h)` was called.
    CaptureStart(CameraHandle),
    /// `queue_frame(h, slot, cb)` was called.
    QueueFrame(CameraHandle, FrameSlotId, FrameCallbackId),
    /// `capture_end(h)` was called.
    CaptureEnd(CameraHandle),
    /// `capture_queue_flush(h)` was called.
    CaptureQueueFlush(CameraHandle),
    /// `frame_revoke_all(h)` was called.
    FrameRevokeAll(CameraHandle),
    /// `register_discovery(cb)` was called.
    RegisterDiscovery(DiscoveryCallbackId),
    /// `unregister_discovery(h)` was called.
    UnregisterDiscovery(DiscoveryRegistrationHandle),
    /// `install_frame_callback` returned `id`.
    InstallFrameCallback(FrameCallbackId),
    /// `uninstall_frame_callback(id)` was called.
    UninstallFrameCallback(FrameCallbackId),
    /// `install_discovery_callback` returned `id`.
    InstallDiscoveryCallback(DiscoveryCallbackId),
    /// `uninstall_discovery_callback(id)` was called.
    UninstallDiscoveryCallback(DiscoveryCallbackId),
}

/// Knob naming the method a caller wants to fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    Startup,
    ListCameras,
    OpenCamera,
    LoadSettings,
    GetFeatureFloat,
    SetFeatureFloat,
    SetFeatureEnum,
    RunFeatureCommand,
    PayloadSize,
    AnnounceFrame,
    CaptureStart,
    QueueFrame,
    RegisterDiscovery,
}

/// Fake behavior configuration.
#[derive(Default)]
struct Config {
    /// `(method, skip)` → error to return on the `skip`-th matching call
    /// (0-indexed). Entries are consumed on use.
    failures: HashMap<(Method, u64), VmbError>,
    /// Per-method invocation counters (matched against `skip`).
    counters: HashMap<Method, u64>,
    /// The camera list returned by `list_cameras`.
    camera_list: Vec<CameraInfo>,
    /// Payload size returned by `payload_size`.
    payload_size: u32,
}

struct State {
    started: AtomicBool,
    config: Mutex<Config>,
    calls: Mutex<Vec<FakeCall>>,
    cameras: Mutex<HashMap<CameraHandle, String>>,
    announced_frames: Mutex<HashMap<FrameSlotId, CameraHandle>>,
    active_captures: Mutex<HashMap<CameraHandle, Vec<(FrameSlotId, FrameCallbackId)>>>,
    frame_callbacks: Mutex<HashMap<FrameCallbackId, Arc<FrameCallback>>>,
    discovery_callbacks: Mutex<HashMap<DiscoveryCallbackId, Arc<DiscoveryCallback>>>,
    discovery_regs: Mutex<HashMap<DiscoveryRegistrationHandle, DiscoveryCallbackId>>,
    counter: AtomicU64,
}

impl State {
    fn new() -> Self {
        Self {
            started: AtomicBool::new(false),
            config: Mutex::new(Config {
                payload_size: 1024,
                ..Config::default()
            }),
            calls: Mutex::new(Vec::new()),
            cameras: Mutex::new(HashMap::new()),
            announced_frames: Mutex::new(HashMap::new()),
            active_captures: Mutex::new(HashMap::new()),
            frame_callbacks: Mutex::new(HashMap::new()),
            discovery_callbacks: Mutex::new(HashMap::new()),
            discovery_regs: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(1),
        }
    }

    fn next_id(&self) -> NonZeroU64 {
        NonZeroU64::new(self.counter.fetch_add(1, Ordering::Relaxed))
            .expect("fake id counter exhausted")
    }

    fn next_u64(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    fn record(&self, call: FakeCall) {
        self.calls.lock().unwrap().push(call);
    }

    /// If the caller has rigged the `n`-th call to this method to fail,
    /// return the injected error. Otherwise return `Ok(())`.
    fn maybe_fail(&self, method: Method) -> Result<()> {
        let mut cfg = self.config.lock().unwrap();
        let counter = cfg.counters.entry(method).or_insert(0);
        let idx = *counter;
        *counter += 1;
        if let Some(err) = cfg.failures.remove(&(method, idx)) {
            return Err(err);
        }
        Ok(())
    }
}

/// In-memory fake implementation of [`VmbRuntime`].
///
/// Cheap to clone — state is `Arc`-shared. Both the `FakeVmbRuntime`
/// held by the runtime-owning component under test AND the clone held
/// by the test harness see the same state.
#[derive(Clone)]
pub struct FakeVmbRuntime {
    state: Arc<State>,
}

impl Default for FakeVmbRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeVmbRuntime {
    /// Construct a fresh fake. `list_cameras` returns `[]`,
    /// `payload_size` returns `1024`, every other method succeeds by
    /// default.
    pub fn new() -> Self {
        Self {
            state: Arc::new(State::new()),
        }
    }

    // --- Programmable knobs -------------------------------------------

    /// Make the `skip`-th future call to `method` return `err`
    /// (0-indexed).
    pub fn fail_nth(&self, method: Method, skip: u64, err: VmbError) {
        self.state
            .config
            .lock()
            .unwrap()
            .failures
            .insert((method, skip), err);
    }

    /// Shortcut for `fail_nth(method, 0, err)`.
    pub fn fail_next(&self, method: Method, err: VmbError) {
        self.fail_nth(method, 0, err);
    }

    /// Set the list of cameras returned by `list_cameras`.
    pub fn set_camera_list(&self, cameras: Vec<CameraInfo>) {
        self.state.config.lock().unwrap().camera_list = cameras;
    }

    /// Set the payload size returned by `payload_size`.
    pub fn set_payload_size(&self, size: u32) {
        self.state.config.lock().unwrap().payload_size = size;
    }

    /// Pre-seed the fake with a camera in the "already open" state
    /// **without** going through the [`VmbRuntime::open_camera`] code
    /// path: no [`FakeCall`] is recorded, failure injection
    /// ([`Self::fail_next`]) is **not** consulted, and no camera-list
    /// or capture-lifecycle side effects are triggered.
    ///
    /// A fresh [`CameraHandle`] is allocated, bound to `id` in the
    /// fake's internal camera map, and returned. After this call
    /// [`Self::handle_for`] returns `Some(handle)` for `id`, and a
    /// subsequent call to [`VmbRuntime::close_camera`] with the
    /// returned handle will drop the binding exactly as for a
    /// normally-opened camera.
    ///
    /// This is a scenario hook for consumer tests that need to drive
    /// production code into the "camera already open" branch of its
    /// own internal guards (e.g. `if state.cameras.contains_key(id)
    /// { return Ok(()) }`) without first running the full open /
    /// capture-setup sequence. The returned handle can be fed into
    /// whatever seam the consumer exposes for pre-populating its own
    /// bookkeeping.
    ///
    /// Calling this multiple times with the same `id` is allowed and
    /// allocates a fresh handle each time; the most recent handle
    /// wins for [`Self::handle_for`] lookups only if earlier handles
    /// are dropped — both bindings coexist in the map otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use vmb_fake::FakeVmbRuntime;
    ///
    /// let fake = FakeVmbRuntime::new();
    /// let handle = fake.pre_open_camera("cam-a");
    /// assert_eq!(fake.handle_for("cam-a"), Some(handle));
    /// // No OpenCamera call was recorded:
    /// assert!(fake.calls().is_empty());
    /// ```
    pub fn pre_open_camera(&self, id: &str) -> CameraHandle {
        let handle = CameraHandle::new(self.state.next_id());
        self.state
            .cameras
            .lock()
            .unwrap()
            .insert(handle, id.to_string());
        handle
    }

    // --- Call log inspection ------------------------------------------

    /// Return a snapshot of every port-method call in order.
    pub fn calls(&self) -> Vec<FakeCall> {
        self.state.calls.lock().unwrap().clone()
    }

    /// Number of recorded calls.
    pub fn call_count(&self) -> usize {
        self.state.calls.lock().unwrap().len()
    }

    /// Whether `startup` has been called and not yet paired with a
    /// `shutdown`.
    pub fn is_started(&self) -> bool {
        self.state.started.load(Ordering::SeqCst)
    }

    /// Look up the handle allocated for the camera with the given ID
    /// string. Returns `None` if the camera is not currently open.
    pub fn handle_for(&self, id: &str) -> Option<CameraHandle> {
        self.state
            .cameras
            .lock()
            .unwrap()
            .iter()
            .find(|(_h, s)| s.as_str() == id)
            .map(|(h, _)| *h)
    }

    // --- Drive callbacks synchronously --------------------------------

    /// Drive every registered discovery callback with `event`.
    pub fn emit_discovery(&self, event: DiscoveryEvent) {
        let callbacks: Vec<Arc<DiscoveryCallback>> = self
            .state
            .discovery_callbacks
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        for cb in callbacks {
            cb.invoke(event.clone());
        }
    }

    /// Drive the frame callback attached to a queued slot on the given
    /// camera. Panics if the capture is not live. Returns `true` if a
    /// frame was delivered, `false` if the camera has no queued slots.
    pub fn deliver_frame(
        &self,
        camera: CameraHandle,
        bytes: &[u8],
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
    ) -> bool {
        let slot_cb = {
            let active = self.state.active_captures.lock().unwrap();
            active.get(&camera).and_then(|slots| slots.first().copied())
        };
        let Some((_slot, cb_id)) = slot_cb else {
            return false;
        };
        let cb = self
            .state
            .frame_callbacks
            .lock()
            .unwrap()
            .get(&cb_id)
            .cloned();
        let Some(cb) = cb else { return false };

        let frame_id = self.state.next_u64();
        let ts_ns = frame_id; // deterministic for tests
        let frame = Frame::new(bytes, width, height, pixel_format, ts_ns, frame_id);
        cb.invoke(&frame);
        true
    }
}

impl VmbRuntime for FakeVmbRuntime {
    // --- Lifecycle ----------------------------------------------------

    fn startup(&self) -> Result<()> {
        self.state.record(FakeCall::Startup);
        self.state.maybe_fail(Method::Startup)?;
        if self
            .state
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(VmbError::AlreadyStarted);
        }
        Ok(())
    }

    fn shutdown(&self) {
        self.state.record(FakeCall::Shutdown);
        self.state.started.store(false, Ordering::SeqCst);
    }

    // --- Cameras ------------------------------------------------------

    fn list_cameras(&self) -> Result<Vec<CameraInfo>> {
        self.state.record(FakeCall::ListCameras);
        self.state.maybe_fail(Method::ListCameras)?;
        Ok(self.state.config.lock().unwrap().camera_list.clone())
    }

    fn open_camera(&self, id: &str) -> Result<CameraHandle> {
        self.state.record(FakeCall::OpenCamera(id.to_string()));
        self.state.maybe_fail(Method::OpenCamera)?;
        let h = CameraHandle::new(self.state.next_id());
        self.state.cameras.lock().unwrap().insert(h, id.to_string());
        Ok(h)
    }

    fn close_camera(&self, h: CameraHandle) {
        self.state.record(FakeCall::CloseCamera(h));
        self.state.cameras.lock().unwrap().remove(&h);
        self.state.active_captures.lock().unwrap().remove(&h);
    }

    fn load_settings(&self, h: CameraHandle, path: &Path) -> Result<()> {
        self.state
            .record(FakeCall::LoadSettings(h, path.to_path_buf()));
        self.state.maybe_fail(Method::LoadSettings)
    }

    fn get_feature_float(&self, h: CameraHandle, name: &str) -> Result<f64> {
        self.state
            .record(FakeCall::GetFeatureFloat(h, name.to_string()));
        self.state.maybe_fail(Method::GetFeatureFloat)?;
        Ok(0.0)
    }

    fn set_feature_float(&self, h: CameraHandle, name: &str, value: f64) -> Result<()> {
        self.state.record(FakeCall::SetFeatureFloat(
            h,
            name.to_string(),
            value.to_bits(),
        ));
        self.state.maybe_fail(Method::SetFeatureFloat)
    }

    fn set_feature_enum(&self, h: CameraHandle, name: &str, value: &str) -> Result<()> {
        self.state.record(FakeCall::SetFeatureEnum(
            h,
            name.to_string(),
            value.to_string(),
        ));
        self.state.maybe_fail(Method::SetFeatureEnum)
    }

    fn run_feature_command(&self, h: CameraHandle, name: &str) -> Result<()> {
        self.state
            .record(FakeCall::RunFeatureCommand(h, name.to_string()));
        self.state.maybe_fail(Method::RunFeatureCommand)
    }

    // --- Capture ------------------------------------------------------

    fn payload_size(&self, h: CameraHandle) -> Result<u32> {
        self.state.record(FakeCall::PayloadSize(h));
        self.state.maybe_fail(Method::PayloadSize)?;
        Ok(self.state.config.lock().unwrap().payload_size)
    }

    fn announce_frame(&self, h: CameraHandle, size: u32) -> Result<FrameSlotId> {
        self.state.record(FakeCall::AnnounceFrame(h, size));
        self.state.maybe_fail(Method::AnnounceFrame)?;
        let slot = FrameSlotId(self.state.next_u64());
        self.state.announced_frames.lock().unwrap().insert(slot, h);
        Ok(slot)
    }

    fn capture_start(&self, h: CameraHandle) -> Result<()> {
        self.state.record(FakeCall::CaptureStart(h));
        self.state.maybe_fail(Method::CaptureStart)?;
        self.state
            .active_captures
            .lock()
            .unwrap()
            .entry(h)
            .or_default();
        Ok(())
    }

    fn queue_frame(&self, h: CameraHandle, slot: FrameSlotId, cb: FrameCallbackId) -> Result<()> {
        self.state.record(FakeCall::QueueFrame(h, slot, cb));
        self.state.maybe_fail(Method::QueueFrame)?;
        self.state
            .active_captures
            .lock()
            .unwrap()
            .entry(h)
            .or_default()
            .push((slot, cb));
        Ok(())
    }

    fn capture_end(&self, h: CameraHandle) {
        self.state.record(FakeCall::CaptureEnd(h));
        self.state.active_captures.lock().unwrap().remove(&h);
    }

    fn capture_queue_flush(&self, h: CameraHandle) {
        self.state.record(FakeCall::CaptureQueueFlush(h));
    }

    fn frame_revoke_all(&self, h: CameraHandle) {
        self.state.record(FakeCall::FrameRevokeAll(h));
        let mut announced = self.state.announced_frames.lock().unwrap();
        announced.retain(|_slot, owner| *owner != h);
    }

    // --- Discovery ----------------------------------------------------

    fn register_discovery(&self, cb: DiscoveryCallbackId) -> Result<DiscoveryRegistrationHandle> {
        self.state.record(FakeCall::RegisterDiscovery(cb));
        self.state.maybe_fail(Method::RegisterDiscovery)?;
        let h = DiscoveryRegistrationHandle(self.state.next_u64());
        self.state.discovery_regs.lock().unwrap().insert(h, cb);
        Ok(h)
    }

    fn unregister_discovery(&self, r: DiscoveryRegistrationHandle) {
        self.state.record(FakeCall::UnregisterDiscovery(r));
        self.state.discovery_regs.lock().unwrap().remove(&r);
    }

    // --- Callback installation ----------------------------------------

    fn install_frame_callback(&self, cb: Arc<FrameCallback>) -> FrameCallbackId {
        let id = FrameCallbackId(self.state.next_u64());
        self.state.record(FakeCall::InstallFrameCallback(id));
        self.state.frame_callbacks.lock().unwrap().insert(id, cb);
        id
    }

    fn uninstall_frame_callback(&self, id: FrameCallbackId) {
        self.state.record(FakeCall::UninstallFrameCallback(id));
        self.state.frame_callbacks.lock().unwrap().remove(&id);
    }

    fn install_discovery_callback(&self, cb: Arc<DiscoveryCallback>) -> DiscoveryCallbackId {
        let id = DiscoveryCallbackId(self.state.next_u64());
        self.state.record(FakeCall::InstallDiscoveryCallback(id));
        self.state
            .discovery_callbacks
            .lock()
            .unwrap()
            .insert(id, cb);
        id
    }

    fn uninstall_discovery_callback(&self, id: DiscoveryCallbackId) {
        self.state.record(FakeCall::UninstallDiscoveryCallback(id));
        self.state.discovery_callbacks.lock().unwrap().remove(&id);
    }
}
