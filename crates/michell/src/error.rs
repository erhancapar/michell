//! The engine's error type: a stable code for programs, a message for people, and the numbers involved.
//!
//! The engine never panics on bad input; it returns one of these instead. The Python binding turns it
//! into `michell.EngineError` with the same three fields.

use std::fmt;

/// What went wrong, as a code that never changes once published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    /// An argument is wrong: a length, a range, a value that is not a number.
    InvalidInput,
    /// The stiffness matrix cannot be factored: the part is not held against every rigid motion.
    SingularSystem,
    /// A bug in michell (for example a caught panic). Never the user's fault.
    InternalError,
}

impl ErrorCode {
    /// The code as text, the form Python and agents see: `"invalid_input"`, `"singular_system"`, ...
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::InvalidInput => "invalid_input",
            ErrorCode::SingularSystem => "singular_system",
            ErrorCode::InternalError => "internal_error",
        }
    }
}

/// One number attached to an error, kept as an integer when it is a count or an index.
#[derive(Clone, Debug, PartialEq)]
pub enum DetailValue {
    Integer(i64),
    Number(f64),
}

/// An error returned by the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineError {
    pub code: ErrorCode,
    pub message: String,
    /// The numbers involved, by name, in the order they were added.
    pub details: Vec<(String, DetailValue)>,
}

impl EngineError {
    /// An error with a code and a message and no numbers yet.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> EngineError {
        EngineError {
            code,
            message: message.into(),
            details: Vec::new(),
        }
    }

    /// Shorthand for an `invalid_input` error.
    pub fn invalid_input(message: impl Into<String>) -> EngineError {
        EngineError::new(ErrorCode::InvalidInput, message)
    }

    /// Adds an integer detail, such as a length or an index.
    pub fn with_integer(mut self, name: &str, value: usize) -> EngineError {
        // Lengths and indices in this engine stay far below i64::MAX.
        let value = i64::try_from(value).unwrap_or(i64::MAX);
        self.details
            .push((name.to_string(), DetailValue::Integer(value)));
        self
    }

    /// Adds a real-number detail, such as a pivot or a modulus.
    pub fn with_number(mut self, name: &str, value: f64) -> EngineError {
        self.details
            .push((name.to_string(), DetailValue::Number(value)));
        self
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_have_their_published_names() {
        assert_eq!(ErrorCode::InvalidInput.as_str(), "invalid_input");
        assert_eq!(ErrorCode::SingularSystem.as_str(), "singular_system");
        assert_eq!(ErrorCode::InternalError.as_str(), "internal_error");
    }

    #[test]
    fn details_keep_their_order_and_kind() {
        let error = EngineError::invalid_input("rho has the wrong length")
            .with_integer("expected", 6)
            .with_number("got", 2.5);
        assert_eq!(
            error.details[0],
            ("expected".to_string(), DetailValue::Integer(6))
        );
        assert_eq!(
            error.details[1],
            ("got".to_string(), DetailValue::Number(2.5))
        );
        assert_eq!(error.to_string(), "invalid_input: rho has the wrong length");
    }
}
