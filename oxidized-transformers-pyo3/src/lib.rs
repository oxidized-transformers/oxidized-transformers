use pyo3::{pymodule, types::PyModule, Bound, PyResult};

#[pymodule]
fn oxidized_transformers(_m: &Bound<PyModule>) -> PyResult<()> {
    Ok(())
}
