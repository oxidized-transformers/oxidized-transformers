use super::pieces::PiecesWithIds;
use crate::{error::BoxedError, repository::repo::Repo};

/// Input for encoding with a tokenizer.
pub enum TokenizerEncodeInput<I>
where
    I: AsRef<str>,
{
    RawString(I),
    // TODO: add chunked input
}

impl From<String> for TokenizerEncodeInput<String> {
    fn from(s: String) -> Self {
        TokenizerEncodeInput::RawString(s)
    }
}

impl From<&str> for TokenizerEncodeInput<String> {
    fn from(s: &str) -> Self {
        TokenizerEncodeInput::RawString(s.to_owned())
    }
}

/// Trait implemented by all tokenizers.
pub trait Tokenizer {
    /// Split one or more texts into pieces.
    ///
    /// * input - Sequences to tokenize. If the sequences are
    ///   strings, they are automatically converted to chunks.
    ///
    /// Returns: Pieces in each sequence.
    fn encode<V, I>(&self, input: V) -> Result<PiecesWithIds, BoxedError>
    where
        V: AsRef<[TokenizerEncodeInput<I>]>,
        I: AsRef<str>;

    /// Reconstruct string sequences from piece identifiers.
    ///
    /// * input - The piece identifiers to reconstruct the strings from.
    /// * skip_special_pieces - Skip special pieces during decoding.
    ///
    /// Returns: The decoded strings.
    fn decode<V, I>(&self, input: V, skip_special_pieces: bool) -> Result<Vec<String>, BoxedError>
    where
        V: AsRef<[I]>,
        I: AsRef<[u32]>;

    /// Get the ID for a single piece.
    ///
    /// * piece - The piece to look up the identifier for.
    ///
    /// Returns: The piece identifier, `None` when the piece
    /// is unknown.
    fn piece_to_id(&self, piece: impl AsRef<str>) -> Option<u32>;

    /// Get the end-of-sequence piece.
    ///
    /// Returns: The end-of-sequence piece or
    /// `None` when this piece is not defined.
    fn eos_piece(&self) -> Option<&str>;
}

/// Trait implemented by tokenizers that can be loaded from a repository.
pub trait FromRepo
where
    Self: Sized + Tokenizer,
{
    /// Load a tokenizer from a repository.
    ///
    /// * repo - The repository to load the tokenizer from.
    ///
    /// Returns: The tokenizer loaded from the repository.
    fn from_repo(repo: &impl Repo) -> Result<Self, BoxedError>;
}
