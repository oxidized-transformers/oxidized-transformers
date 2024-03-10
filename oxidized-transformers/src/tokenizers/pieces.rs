use candle_core::{Device, Tensor};
use snafu::{OptionExt, ResultExt, Snafu};

use crate::layers::attention::{AttentionMask, AttentionMaskError};

/// `PiecesWithIds` errors.
#[derive(Debug, Snafu)]
pub enum PiecesWithIdsError {
    #[snafu(display("Cannot calculate maximum sequence length"))]
    MaxLength,

    #[snafu(display("Cannot create padded tensor"))]
    PaddedTensor { source: candle_core::Error },

    #[snafu(display("Cannot create boolean mask"))]
    BoolMask { source: candle_core::Error },

    #[snafu(display("Cannot create attention mask"))]
    AttentionMask { source: AttentionMaskError },
}

/// Encoded output of tokenizers.
#[derive(Debug, Clone, Default)]
pub struct PiecesWithIds {
    /// Piece identifiers of each input sequence.
    pub ids: Vec<Vec<u32>>,
    /// Piece strings of each input sequence.
    pub pieces: Vec<Vec<String>>,
}

impl PiecesWithIds {
    /// Generate a padded tensor of the piece identifiers.
    ///
    /// * padding_id - Piece identifier of the padding piece. The actual identifier
    ///   generally doesn't matter when an attention mask is used (and
    ///   as long as it is a valid vocabulary index).
    /// * pad_left - When `false`, sequences shorter than the longest sequence are
    ///   right-padded. Otherwise, sequences are left-padding..
    /// * device - Device on which the padded tensor is created.
    ///
    /// Returns: The padded piece ids.
    /// *Shape:* ``(batch_size, max_seq_len)``
    pub fn padded_tensor(
        &self,
        padding_id: u32,
        pad_left: bool,
        device: &Device,
    ) -> Result<Tensor, PiecesWithIdsError> {
        let n_sequences = self.ids.len();
        let max_len = self
            .ids
            .iter()
            .map(|ids| ids.len())
            .max()
            .context(MaxLengthSnafu)?;

        let mut padded = vec![padding_id; n_sequences * max_len];
        for (i, ids) in self.ids.iter().enumerate() {
            let len = ids.len();
            let start = if pad_left { max_len - len } else { 0 };
            let end = if pad_left { max_len } else { len };
            padded[i * max_len + start..i * max_len + end].copy_from_slice(ids);
        }

        Tensor::from_vec(padded, (n_sequences, max_len), device).context(PaddedTensorSnafu)
    }

    /// Generate the attention masks. The mask is equivalent to:
    /// `ids.padded_tensor(padding_id) != padding_id`
    ///
    /// * pad_left - When `false`, sequences shorter than the longest sequence are
    ///   right-padded. Otherwise, sequences are left-padding..
    /// * device - Device on which the padded tensor is created.
    ///
    /// Returns: The attention mask.
    pub fn attention_mask(
        &self,
        pad_left: bool,
        device: &Device,
    ) -> Result<AttentionMask, PiecesWithIdsError> {
        let n_sequences = self.ids.len();
        let max_len = self
            .ids
            .iter()
            .map(|ids| ids.len())
            .max()
            .context(MaxLengthSnafu)?;

        let mut padded = vec![0u32; n_sequences * max_len];
        for (i, ids) in self.ids.iter().enumerate() {
            let len = ids.len();
            let start = if pad_left { max_len - len } else { 0 };
            let end = if pad_left { max_len } else { len };
            padded[i * max_len + start..i * max_len + end].fill(1u32);
        }

        let bool_mask =
            Tensor::from_vec(padded, (n_sequences, max_len), device).context(BoolMaskSnafu)?;

        AttentionMask::new(bool_mask).context(AttentionMaskSnafu)
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use snafu::{Report, Whatever};

    use super::*;

    #[fixture]
    fn pieces() -> PiecesWithIds {
        PiecesWithIds {
            ids: vec![
                vec![101, 7592, 2088, 102],
                vec![101, 2023, 2003, 1996, 5409, 1999, 1996, 2088, 102],
            ],
            pieces: vec![
                vec![
                    "[CLS]".to_owned(),
                    "hello".to_owned(),
                    "world".to_owned(),
                    "[SEP]".to_owned(),
                ],
                vec![
                    "[CLS]".to_owned(),
                    "this".to_owned(),
                    "is".to_owned(),
                    "the".to_owned(),
                    "worst".to_owned(),
                    "in".to_owned(),
                    "the".to_owned(),
                    "world".to_owned(),
                    "[SEP]".to_owned(),
                ],
            ],
        }
    }

    #[rstest]
    fn piecewithids_attention_mask(pieces: PiecesWithIds) -> Report<Whatever> {
        Report::capture(|| {
            let mask = pieces
                .attention_mask(false, &Device::Cpu)
                .expect("Cannot create attention mask");

            assert_eq!(
                mask.bool_mask()
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2")
                    .as_slice(),
                &[
                    vec![1, 1, 1, 1, 0, 0, 0, 0, 0],
                    vec![1, 1, 1, 1, 1, 1, 1, 1, 1]
                ]
            );

            let mask = pieces
                .attention_mask(true, &Device::Cpu)
                .expect("Cannot create attention mask");

            assert_eq!(
                mask.bool_mask()
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2")
                    .as_slice(),
                &[
                    vec![0, 0, 0, 0, 0, 1, 1, 1, 1,],
                    vec![1, 1, 1, 1, 1, 1, 1, 1, 1]
                ]
            );

            Ok(())
        })
    }

    #[rstest]
    fn piecewithids_padded_tensor(pieces: PiecesWithIds) -> Report<Whatever> {
        Report::capture(|| {
            let padding_id = u32::MAX;

            let padded = pieces
                .padded_tensor(padding_id, false, &Device::Cpu)
                .expect("Cannot create padded tensor");

            assert_eq!(
                padded
                    .to_vec2::<u32>()
                    .expect("Cannot convert to vec2")
                    .as_slice(),
                &[
                    vec![
                        101, 7592, 2088, 102, padding_id, padding_id, padding_id, padding_id,
                        padding_id
                    ],
                    vec![101, 2023, 2003, 1996, 5409, 1999, 1996, 2088, 102]
                ]
            );

            let padded = pieces
                .padded_tensor(padding_id, true, &Device::Cpu)
                .expect("Cannot create padded tensor");

            assert_eq!(
                padded
                    .to_vec2::<u32>()
                    .expect("Cannot convert to vec2")
                    .as_slice(),
                &[
                    vec![
                        padding_id, padding_id, padding_id, padding_id, padding_id, 101, 7592,
                        2088, 102
                    ],
                    vec![101, 2023, 2003, 1996, 5409, 1999, 1996, 2088, 102]
                ]
            );

            Ok(())
        })
    }
}
