use std::path::Path;
use std::{fs::File, path::PathBuf};

use snafu::{ensure, ResultExt, Snafu};
use tokenizers::tokenizer::Tokenizer as HuggingFaceTokenizer;

use super::pieces::PiecesWithIds;
use super::tokenizer::FromRepo;
use super::{
    hf_hub::FromHFHub,
    tokenizer::{Tokenizer, TokenizerEncodeInput},
};
use crate::error::BoxedError;
use crate::repository::repo::Repo;

/// `HfTokenizer` errors.
#[derive(Debug, Snafu)]
pub enum HfTokenizerError {
    #[snafu(display("Couldn't encode tokenizer inputs into pieces and ids"))]
    Encode { source: tokenizers::Error },

    #[snafu(display("Couldn't decode piece identifiers into strings"))]
    Decode { source: tokenizers::Error },

    #[snafu(display("Couldn't open 'tokenizer.json'"))]
    OpenTokenizerJSON { source: BoxedError },

    #[snafu(display("'tokenizer.json' file is missing"))]
    MissingTokenizerJSON,

    #[snafu(display("Couldn't open 'tokenizer_config.json'"))]
    OpenTokenizerConfigJSON { source: BoxedError },

    #[snafu(display("Couldn't open 'special_tokens_map.json'"))]
    OpenSpecialTokensMapJSON { source: BoxedError },

    #[snafu(display("Couldn't open JSON file at {}", path.to_string_lossy()))]
    OpenJSON {
        path: PathBuf,
        source: std::io::Error,
    },

    #[snafu(display("Cannot deserialize JSON file at {}", path.to_string_lossy()))]
    DeserializeJSON {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[snafu(display("Couldn't load Hugging Face tokenizer from config"))]
    LoadHFTokenizer { source: BoxedError },
}

/// Wraps the tokenizers from the HuggingFace `tokenizers` package. It supports a
/// wide range of piece tokenizers, including word piece, byte pair encoding, and
/// sentencepiece unigram tokenizers. This is the tokenizer that should be used
/// in the majority of cases
pub struct HfTokenizer {
    tokenizer: HuggingFaceTokenizer,
    eos_piece: Option<String>,
}

impl HfTokenizer {
    fn new(
        tokenizer: HuggingFaceTokenizer,
        config: Option<&config::ConfigWithEosToken>,
        special_tokens_map: Option<&config::ConfigWithEosToken>,
    ) -> Self {
        let eos_piece = config
            .and_then(|e| e.eos_token())
            .or_else(|| special_tokens_map.and_then(|e| e.eos_token()));

        Self {
            tokenizer,
            eos_piece: eos_piece.cloned(),
        }
    }

    fn try_parse_json_config(
        path: &impl AsRef<Path>,
    ) -> Result<Option<config::ConfigWithEosToken>, BoxedError> {
        let file = File::open(path.as_ref()).context(OpenJSONSnafu {
            path: path.as_ref(),
        })?;

        let deserialized: Option<config::ConfigWithEosToken> = serde_json::from_reader(file)
            .context(DeserializeJSONSnafu {
                path: path.as_ref().to_owned(),
            })
            .boxed()?;

        Ok(deserialized)
    }
}

impl Tokenizer for HfTokenizer {
    fn encode<V, I>(&self, input: V) -> Result<PiecesWithIds, BoxedError>
    where
        V: AsRef<[TokenizerEncodeInput<I>]>,
        I: AsRef<str>,
    {
        let converted_input = input
            .as_ref()
            .iter()
            .map(|input| match input {
                TokenizerEncodeInput::RawString(s) => {
                    tokenizers::EncodeInput::Single(s.as_ref().into())
                }
            })
            .collect::<Vec<_>>();

        let encoding = self
            .tokenizer
            .encode_batch(converted_input, true)
            .context(EncodeSnafu)?;

        Ok(PiecesWithIds {
            ids: encoding
                .iter()
                .map(|ids| ids.get_ids().to_owned())
                .collect(),
            pieces: encoding
                .iter()
                .map(|ids| ids.get_tokens().to_owned())
                .collect(),
        })
    }

    fn decode<V, I>(&self, input: V, skip_special_pieces: bool) -> Result<Vec<String>, BoxedError>
    where
        V: AsRef<[I]>,
        I: AsRef<[u32]>,
    {
        let converted_input = input
            .as_ref()
            .iter()
            .map(|input| input.as_ref())
            .collect::<Vec<_>>();

        self.tokenizer
            .decode_batch(&converted_input, skip_special_pieces)
            .context(DecodeSnafu)
            .boxed()
    }

    fn piece_to_id(&self, piece: impl AsRef<str>) -> Option<u32> {
        self.tokenizer.token_to_id(piece.as_ref())
    }

    fn eos_piece(&self) -> Option<&str> {
        self.eos_piece.as_deref()
    }
}

impl FromRepo for HfTokenizer {
    fn from_repo(repo: &impl Repo) -> Result<Self, BoxedError> {
        let tokenizer_json = repo
            .file("tokenizer.json")
            .context(OpenTokenizerJSONSnafu)
            .boxed()?;
        let tokenizer_config_json = repo
            .file("tokenizer_config.json")
            .context(OpenTokenizerConfigJSONSnafu)
            .boxed()?;
        let special_tokens_map_json = repo
            .file("special_tokens_map.json")
            .context(OpenSpecialTokensMapJSONSnafu)
            .boxed()?;

        ensure!(tokenizer_json.is_some(), MissingTokenizerJSONSnafu);
        let tokenizer = HuggingFaceTokenizer::from_file(tokenizer_json.unwrap())
            .context(LoadHFTokenizerSnafu)?;

        let tokenizer_config = tokenizer_config_json
            .map(|p| Self::try_parse_json_config(&p))
            .transpose()?
            .flatten();

        let special_tokens_map = special_tokens_map_json
            .map(|p| Self::try_parse_json_config(&p))
            .transpose()?
            .flatten();

        Ok(Self::new(
            tokenizer,
            tokenizer_config.as_ref(),
            special_tokens_map.as_ref(),
        ))
    }
}

impl FromHFHub for HfTokenizer {}

mod config {
    use std::collections::HashMap;

    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    /// Represents an EOS token in the tokenizer configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(super) enum EosTokenInConfig {
        Default(String),
        Wrapped { content: Option<String> },
    }

    /// Represents a tokenizer configuration that includes an EOS token.
    /// Primarily used to with `tokenizer_config.json` and `special_tokens_map.json` files.
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub(super) struct ConfigWithEosToken {
        #[serde(default)]
        eos_token: Option<EosTokenInConfig>,
        #[serde(flatten)]
        _extra: HashMap<String, Value>,
    }

    impl ConfigWithEosToken {
        pub(crate) fn eos_token(&self) -> Option<&String> {
            self.eos_token.as_ref().and_then(|e| match e {
                EosTokenInConfig::Default(s) => Some(s),
                EosTokenInConfig::Wrapped { content } => content.as_ref(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use rstest::{fixture, rstest};
    use tokenizers::{tokenizer::Tokenizer as HuggingFaceTokenizer, EncodeInput, PaddingParams};

    use super::*;

    #[fixture]
    fn short_sample_texts() -> &'static [&'static str] {
        &[
            "I saw a girl with a telescope.",
            "Today we will eat poké bowl, lots of it!",
            "Tokens which are unknown inペ mostで latinが alphabet際 vocabularies.",
        ]
    }

    #[fixture]
    fn long_sample_texts() -> &'static [&'static str] {
        // Two short Wikipedia fragments from:
        // https://en.wikipedia.org/wiki/Kinesis_(keyboard)#Contoured_/_Advantage
        // https://en.wikipedia.org/wiki/Doom_(1993_video_game)#Engine
        &[
            r#"The original Model 100, released in 1992, featured a single-piece 
        "contoured design similar to the Maltron keyboard, with the keys laid 
        "out in a traditional QWERTY arrangement, separated into two clusters
        "for the left and right hands.[2] A 1993 article in PC Magazine 
        "described the US$690 (equivalent to $1,300 in 2021) keyboard's
        'arrangement as having "the alphabet keys in precisely vertical
        "(not diagonal) columns in two concave depressions. The Kinesis
        "Keyboard also puts the Backspace, Delete, Enter, Space, Ctrl, Alt,
        "Home, End, Page Up, and Page Down keys under your thumbs in the 
        'middle.[23]"#,
            r#"Doom was programmed largely in the ANSI C programming language, with "
        "a few elements in assembly language. Development was done on NeXT "
        "computers running the NeXTSTEP operating system.[35] The data used by "
        "the game engine, including level designs and graphics files, are "
        'stored in WAD files, short for "Where\'s All the Data?"."#,
        ]
    }

    fn compare_tokenizer_outputs_with_hf_tokenizer(
        model_name: &str,
        pad_token: Option<&str>,
        eos_piece: Option<&str>,
        texts: &[&str],
    ) {
        let tokenizer = HfTokenizer::from_hf_hub(model_name, None)
            .expect("Failed to load tokenizer from HF Hub");
        let mut hf_tokenizer = HuggingFaceTokenizer::from_pretrained(model_name, None)
            .expect("Failed to load HF tokenizer from HF Hub");

        assert_eq!(tokenizer.eos_piece(), eos_piece.as_deref());

        let our_input: Vec<TokenizerEncodeInput<_>> = texts.iter().map(|s| (*s).into()).collect();
        let hf_input: Vec<EncodeInput> = texts.iter().map(|s| (*s).into()).collect();

        let mut right_padding = PaddingParams::default();
        right_padding.pad_token = pad_token
            .unwrap_or(right_padding.pad_token.as_ref())
            .to_string();
        let mut left_padding = PaddingParams::default();
        left_padding.direction = tokenizers::PaddingDirection::Left;
        left_padding.pad_token = pad_token
            .unwrap_or(right_padding.pad_token.as_ref())
            .to_string();

        let our_encoded = tokenizer.encode(our_input).expect("Failed to encode input");

        // Right padding.
        let our_encoded_padded_right = our_encoded
            .padded_tensor(right_padding.pad_id, false, &Device::Cpu)
            .expect("Failed to pad tensor")
            .to_vec2::<u32>()
            .expect("Failed to convert tensor to vec2");
        let our_encoded_attn_mask_padded_right = our_encoded
            .attention_mask(false, &Device::Cpu)
            .expect("Cannot create attention mask");
        let our_encoded_attn_mask_padded_right =
            match our_encoded_attn_mask_padded_right.bool_mask().dims2() {
                Ok((_, _)) => our_encoded_attn_mask_padded_right
                    .bool_mask()
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2"),
                _ => our_encoded_attn_mask_padded_right
                    .bool_mask()
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2"),
            };
        let hf_encoded_padded_right = hf_tokenizer
            .with_padding(Some(right_padding.clone()))
            .encode_batch(hf_input.clone(), true)
            .expect("Failed to encode input");

        for (ours, hf) in our_encoded_padded_right
            .iter()
            .zip(hf_encoded_padded_right.iter())
        {
            assert_eq!(ours.as_slice(), hf.get_ids());
        }

        for (ours, hf) in our_encoded_attn_mask_padded_right
            .iter()
            .zip(hf_encoded_padded_right.iter())
        {
            assert_eq!(ours.as_slice(), hf.get_attention_mask());
        }

        // Left padding.
        let our_encoded_padded_left = our_encoded
            .padded_tensor(left_padding.pad_id, true, &Device::Cpu)
            .expect("Failed to pad tensor")
            .to_vec2::<u32>()
            .expect("Failed to convert tensor to vec2");
        let our_encoded_attn_mask_padded_left = our_encoded
            .attention_mask(true, &Device::Cpu)
            .expect("Cannot create attention mask");
        let our_encoded_attn_mask_padded_left =
            match our_encoded_attn_mask_padded_left.bool_mask().dims2() {
                Ok((_, _)) => our_encoded_attn_mask_padded_left
                    .bool_mask()
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2"),
                _ => our_encoded_attn_mask_padded_left
                    .bool_mask()
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .squeeze(1)
                    .expect("Failed to squeeze attn mask")
                    .to_vec2::<u32>()
                    .expect("Cannot convert mask to vec2"),
            };
        let hf_encoded_padded_left = hf_tokenizer
            .with_padding(Some(left_padding.clone()))
            .encode_batch(hf_input.clone(), true)
            .expect("Failed to encode input");

        for (ours, hf) in our_encoded_padded_left
            .iter()
            .zip(hf_encoded_padded_left.iter())
        {
            assert_eq!(ours.as_slice(), hf.get_ids());
        }

        for (ours, hf) in our_encoded_attn_mask_padded_left
            .iter()
            .zip(hf_encoded_padded_left.iter())
        {
            assert_eq!(ours.as_slice(), hf.get_attention_mask());
        }

        // Decoding.
        let our_decoded = tokenizer
            .decode(our_encoded.ids.iter(), true)
            .expect("Failed to decode input");
        let hf_decoded = hf_tokenizer
            .with_padding(Some(right_padding.clone()))
            .decode_batch(
                hf_encoded_padded_right
                    .iter()
                    .map(|v| v.get_ids())
                    .collect::<Vec<_>>()
                    .as_slice(),
                true,
            )
            .expect("Failed to decode input");

        assert_eq!(our_decoded, hf_decoded);
    }

    #[rstest]
    #[case("bert-base-cased", None, None)]
    #[case("camembert-base", None, None)]
    #[case("roberta-base", None, None)]
    #[case("xlm-roberta-base", None, None)]
    #[case("EleutherAI/gpt-neox-20b", Some("[PAD]"), Some("<|endoftext|>"))]
    #[case("ausboss/llama-30b-supercot", Some("</s>"), Some("</s>"))]
    #[case("tiiuae/falcon-7b", Some("<|endoftext|>"), Some("<|endoftext|>"))]
    fn tokenizer_test_against_hugging_face_short(
        #[case] model_name: &str,
        #[case] pad_token: Option<&str>,
        #[case] eos_piece: Option<&str>,
        short_sample_texts: &[&str],
    ) {
        compare_tokenizer_outputs_with_hf_tokenizer(
            model_name,
            pad_token,
            eos_piece,
            short_sample_texts,
        );
    }

    #[rstest]
    #[case("bert-base-cased", None, None)]
    #[case("camembert-base", None, None)]
    #[case("roberta-base", None, None)]
    #[case("xlm-roberta-base", None, None)]
    #[case("EleutherAI/gpt-neox-20b", Some("[PAD]"), Some("<|endoftext|>"))]
    #[case("ausboss/llama-30b-supercot", Some("</s>"), Some("</s>"))]
    #[case("tiiuae/falcon-7b", Some("<|endoftext|>"), Some("<|endoftext|>"))]
    fn tokenizer_test_against_hugging_face_long(
        #[case] model_name: &str,
        #[case] pad_token: Option<&str>,
        #[case] eos_piece: Option<&str>,
        short_sample_texts: &[&str],
    ) {
        compare_tokenizer_outputs_with_hf_tokenizer(
            model_name,
            pad_token,
            eos_piece,
            short_sample_texts,
        );
    }
}
