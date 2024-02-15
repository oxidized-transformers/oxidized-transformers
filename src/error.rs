use std::error::Error;

/// Alias for boxed errors that can be sent across threads.
pub type BoxedError = Box<dyn Error + Send + Sync>;
