//! Operator implementations.
//!
//! One module per operator. Each owns its state and implements
//! [`crate::operator::Operator`]. The pipeline composes them.

pub mod beats;
pub mod filter;
pub mod meta_talk;
pub mod scenes;
pub mod segment;
pub mod transcription;
pub mod vad;
