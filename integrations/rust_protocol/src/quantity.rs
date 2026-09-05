use serde_json::Number;

const NATIVE_DECIMAL_PLACES: u32 = 12;

/// Exact u64 lots shared by the two native Rust adapters.
/// Output precision never changes the quantity submitted to either engine.
#[derive(Clone)]
pub struct QuantityEncoding {
    output_decimal_places: u32,
}

impl QuantityEncoding {
    pub fn new(output_decimal_places: u32) -> Result<Self, String> {
        if output_decimal_places > 18 {
            return Err("quantity_decimal_places must be between 0 and 18".to_string());
        }
        Ok(Self {
            output_decimal_places,
        })
    }

    pub fn units(&self, value: &Number) -> Result<u64, String> {
        let text = value.to_string();
        if text.starts_with('-') {
            return Err("quantity must be positive".to_string());
        }
        let range_error = || "quantity exceeds the adapter's integer range".to_string();
        let (mantissa, exponent) = match text.split_once(['e', 'E']) {
            Some((mantissa, exponent)) => (
                mantissa,
                exponent.parse::<i64>().map_err(|_| range_error())?,
            ),
            None => (text.as_str(), 0),
        };
        let fractional_places = mantissa.split_once('.').map_or(0, |(_, part)| part.len());
        let digits = mantissa.replace('.', "");
        let significant = digits.trim_start_matches('0');
        if significant.is_empty() {
            return Err("quantity must be positive".to_string());
        }
        let coefficient = significant.trim_end_matches('0');
        let trailing_zeros = significant.len() - coefficient.len();
        let shift = exponent
            .checked_sub(i64::try_from(fractional_places).map_err(|_| range_error())?)
            .and_then(|value| value.checked_add(i64::from(NATIVE_DECIMAL_PLACES)))
            .and_then(|value| value.checked_add(i64::try_from(trailing_zeros).ok()?))
            .ok_or_else(range_error)?;
        if shift < 0 {
            return Err(format!(
                "quantity requires more than {NATIVE_DECIMAL_PLACES} native decimal places"
            ));
        }
        // Parse only the significant digits after removing exact decimal zeros.
        // No binary float or fixed-precision decimal parser may round this input.
        let coefficient = coefficient.parse::<u64>().map_err(|_| range_error())?;
        let multiplier = u32::try_from(shift)
            .ok()
            .and_then(|shift| 10_u64.checked_pow(shift))
            .ok_or_else(range_error)?;
        coefficient.checked_mul(multiplier).ok_or_else(range_error)
    }

    pub fn format(&self, units: u64) -> String {
        let places = self.output_decimal_places.min(NATIVE_DECIMAL_PLACES);
        let divisor = 10_u64.pow(NATIVE_DECIMAL_PLACES - places);
        let mut rounded = units / divisor;
        let remainder = units % divisor;
        if divisor > 1
            && (remainder > divisor / 2 || (remainder == divisor / 2 && rounded % 2 == 1))
        {
            // With divisor >= 10, incrementing the quotient cannot overflow.
            rounded += 1;
        }
        let scale = 10_u64.pow(places);
        let whole = rounded / scale;
        let fraction = rounded % scale;
        if fraction == 0 {
            return whole.to_string();
        }
        format!("{whole}.{fraction:0width$}", width = places as usize)
            .trim_end_matches('0')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn number(text: &str) -> Number {
        Number::from_str(text).unwrap()
    }

    #[test]
    fn output_precision_does_not_round_submitted_lots() {
        for places in [0, 3, 12, 18] {
            let encoding = QuantityEncoding::new(places).unwrap();
            assert_eq!(
                encoding.units(&number("1.2345")).unwrap(),
                1_234_500_000_000
            );
            assert_eq!(encoding.units(&number("1e-12")).unwrap(), 1);
        }
    }

    #[test]
    fn scientific_notation_and_insignificant_zeros_remain_exact() {
        let encoding = QuantityEncoding::new(12).unwrap();
        for (text, units) in [
            ("1E+1", 10_000_000_000_000),
            ("1.234e-9", 1_234),
            (
                "1234000000000000000000000000000000000e-36",
                1_234_000_000_000,
            ),
            ("1.000000000000000000000000000000000000", 1_000_000_000_000),
        ] {
            assert_eq!(encoding.units(&number(text)).unwrap(), units, "{text}");
        }
    }

    #[test]
    fn nonrepresentable_inputs_cannot_round_into_the_native_domain() {
        let encoding = QuantityEncoding::new(0).unwrap();
        for text in [
            "0.0000000000011",
            "1e-13",
            "1.00000000000000000000000000001",
            "100000000000000000000000000001e-29",
        ] {
            assert!(
                encoding
                    .units(&number(text))
                    .unwrap_err()
                    .contains("more than 12"),
                "{text}"
            );
        }
    }

    #[test]
    fn positive_u64_range_is_checked_without_truncation() {
        let encoding = QuantityEncoding::new(12).unwrap();
        assert_eq!(
            encoding.units(&number("18446744.073709551615")).unwrap(),
            u64::MAX
        );
        for text in [
            "0",
            "-1",
            "18446744.073709551616",
            "1e99",
            "1e9999999999999999999",
        ] {
            assert!(encoding.units(&number(text)).is_err(), "{text}");
        }
        assert_eq!(encoding.format(u64::MAX), "18446744.073709551615");
    }

    #[test]
    fn observation_rounding_is_half_even_and_canonical() {
        for (places, text, expected) in [
            (3, "1.2345", "1.234"),
            (3, "1.2355", "1.236"),
            (0, "2.5", "2"),
            (0, "3.5", "4"),
            (18, "0.2", "0.2"),
            (12, "1e-12", "0.000000000001"),
        ] {
            let encoding = QuantityEncoding::new(places).unwrap();
            assert_eq!(
                encoding.format(encoding.units(&number(text)).unwrap()),
                expected
            );
        }
        assert!(QuantityEncoding::new(19).is_err());
    }
}
