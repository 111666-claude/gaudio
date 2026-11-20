package gaudio

import (
	"errors"
	"math"
)

var (
	// errInvalidWidth is returned when the width coefficient is negative
	// or not a finite number.
	errInvalidWidth = errors.New("invalid mid/side width: must be a finite, non-negative number")
)

// MidSideEffect applies mid/side (M/S) stereo processing to the segment.
//
// The interleaved stereo data is converted frame by frame into mid (sum) and
// side (difference) signals, the side signal is scaled by width, and the data
// is converted back to left/right:
//
//	mid  = (left + right) / 2
//	side = (left - right) / 2 * width
//	left  = mid + side
//	right = mid - side
//
// width controls the stereo width:
//
//   - width = 1.0 (the default) keeps the original signal bit-for-bit.
//   - width = 0.0 collapses the signal to mono (the mono preview).
//   - 0.0 < width < 1.0 narrows the stereo image.
//   - width > 1.0 widens the stereo image.
//
// When monoPreview is true the segment is folded down to mono regardless of
// width, which is exactly what a mono playback device would reproduce.
//
// The segment length, sample rate and channel count are never changed. Mono
// (and other non-stereo) segments are returned untouched. A trailing,
// unpaired sample on an odd-length buffer is preserved as-is.
//
// To guarantee the result stays inside the allowed -1.0..1.0 range, the whole
// segment is scaled down uniformly only if processing would push a sample
// past the peak of 1.0; segments that already fit are not rescaled, so the
// default width leaves the original signal intact and the existing
// overlay/export chains keep working.
//
// A negative or non-finite width returns an error and leaves the segment
// unchanged.
func (segment *Segment) MidSideEffect(width float64, monoPreview bool) error {
	if math.IsNaN(width) || math.IsInf(width, 0) || width < 0 {
		return errInvalidWidth
	}

	// Mid/side is a stereo-only operation. Mono segments (and any other
	// channel count) are already mono-compatible, so there is nothing to do.
	if segment.Channels != 2 {
		return nil
	}

	// Mono preview is equivalent to removing the side signal entirely.
	if monoPreview {
		width = 0
	}

	// width of 1.0 reconstructs the original signal exactly; skip the work
	// (and any float rounding) so the default path is a true no-op.
	if width == 1.0 {
		return nil
	}

	w := float32(width)
	dataLen := len(segment.Data)

	// Only process complete interleaved frames. If the buffer has an odd
	// length the trailing, unpaired sample is left untouched.
	frameCount := dataLen / 2
	maxPeak := float32(0)

	for frame := 0; frame < frameCount; frame++ {
		i := frame * 2
		left := segment.Data[i]
		right := segment.Data[i+1]

		mid := (left + right) * 0.5
		side := (left - right) * 0.5 * w

		newLeft := mid + side
		newRight := mid - side
		segment.Data[i] = newLeft
		segment.Data[i+1] = newRight

		if peak := absFloat32(newLeft); peak > maxPeak {
			maxPeak = peak
		}
		if peak := absFloat32(newRight); peak > maxPeak {
			maxPeak = peak
		}
	}

	// Peak protection: never let processing raise the signal above the
	// allowed range. Scale uniformly (preserving the width/energy ratio) and
	// only when strictly necessary. A silent segment has a zero peak and
	// needs no scaling.
	if maxPeak > 1.0 {
		gain := 1.0 / maxPeak
		for frame := 0; frame < frameCount; frame++ {
			i := frame * 2
			segment.Data[i] *= gain
			segment.Data[i+1] *= gain
		}
	}

	return nil
}

// MonoPreview folds the segment down to mono using mid/side processing,
// simulating how the audio sounds on a single-speaker (mono) device.
// It is equivalent to MidSideEffect(0, true).
func (segment *Segment) MonoPreview() error {
	return segment.MidSideEffect(0, true)
}

func absFloat32(value float32) float32 {
	if value < 0 {
		return -value
	}
	return value
}
