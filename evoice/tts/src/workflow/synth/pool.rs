use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError};

/// Fixed set of exclusive workers (onnxruntime sessions run through `&mut`): one per live stream.
#[derive(Debug)]
pub struct Pool<T> {
    free: Mutex<Vec<T>>,
}

/// A worker held by one stream; it returns to its pool when dropped.
#[derive(Debug)]
pub struct Lease<T> {
    worker: Option<T>,
    pool: Arc<Pool<T>>,
}

impl<T> Pool<T> {
    #[must_use]
    pub fn new(workers: Vec<T>) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(workers),
        })
    }

    /// `None` when every worker is busy.
    #[must_use]
    pub fn lease(self: &Arc<Self>) -> Option<Lease<T>> {
        let worker = self.free.lock().unwrap_or_else(PoisonError::into_inner).pop()?;
        Some(Lease {
            worker: Some(worker),
            pool: Arc::clone(self),
        })
    }

    #[must_use]
    pub fn idle(&self) -> usize {
        self.free.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.pool
                .free
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(worker);
        }
    }
}

impl<T> Deref for Lease<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // A lease holds its worker until `drop`; the `None` arm cannot run.
        self.worker.as_ref().unwrap_or_else(|| std::process::abort())
    }
}

impl<T> DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.worker.as_mut().unwrap_or_else(|| std::process::abort())
    }
}
