package gaudio

import (
	"bytes"
	"math"
	"testing"

	"github.com/mjibson/go-dsp/wav"
)

const midsideEpsilon = 1e-6

func approxEqual(a, b float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= midsideEpsilon
}

func approxSliceEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d samples, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEqual(got[i], want[i]) {
			t.Fatalf("sample %d mismatch: got %v, want %v", i, got[i], want[i])
		}
	}
}

func energy(data []float32) float64 {
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	return sum
}

func makeStereoSegment(frames ...[2]float32) *Segment {
	data := make([]float32, 0, len(frames)*2)
	for _, frame := range frames {
		data = append(data, frame[0], frame[1])
	}
	return SegmentFromData(data, 2, 48000)
}

// In-phase (left == right) pulses live entirely in the mid signal.
func TestMidSideInPhasePulses(t *testing.T) {
	// Fixed sample: a pair of in-phase pulses, one positive, one negative.
	seg := makeStereoSegment([2]float32{0.5, 0.5}, [2]float32{-0.25, -0.25})
	original := seg.Clone()

	// Default width keeps the original signal bit-for-bit.
	if err := seg.MidSideEffect(1.0, false); err != nil {
		t.Fatalf("default width returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, original.Data)

	// Width 0 collapses to mono: each channel keeps the mid value (l+r)/2.
	seg = makeStereoSegment([2]float32{0.5, 0.5}, [2]float32{-0.25, -0.25})
	if err := seg.MidSideEffect(0.0, false); err != nil {
		t.Fatalf("width 0 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{0.5, 0.5, -0.25, -0.25})

	// Narrowing and widening do not change an in-phase signal: its side is 0.
	seg = makeStereoSegment([2]float32{0.5, 0.5}, [2]float32{-0.25, -0.25})
	if err := seg.MidSideEffect(0.5, false); err != nil {
		t.Fatalf("width 0.5 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, original.Data)

	seg = makeStereoSegment([2]float32{0.5, 0.5}, [2]float32{-0.25, -0.25})
	if err := seg.MidSideEffect(3.0, false); err != nil {
		t.Fatalf("width 3 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, original.Data)
}

// Anti-phase (left == -right) pulses live entirely in the side signal, so a
// mono fold-down must remove them completely. This is the mobile-speaker
// scenario from the bug report.
func TestMidSideAntiPhasePulses(t *testing.T) {
	seg := makeStereoSegment([2]float32{0.5, -0.5}, [2]float32{-0.25, 0.25})
	original := seg.Clone()

	// Default width is identity.
	if err := seg.MidSideEffect(1.0, false); err != nil {
		t.Fatalf("default width returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, original.Data)

	// Mono preview / width 0: the mid is zero, so the signal disappears.
	seg = makeStereoSegment([2]float32{0.5, -0.5}, [2]float32{-0.25, 0.25})
	if err := seg.MidSideEffect(0.0, false); err != nil {
		t.Fatalf("width 0 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{0, 0, 0, 0})

	if e := energy(seg.Data); e != 0 {
		t.Fatalf("anti-phase mono fold-down must be silent, got energy %v", e)
	}

	// Narrowing to half width scales the side by 0.5.
	seg = makeStereoSegment([2]float32{0.5, -0.5}, [2]float32{-0.25, 0.25})
	if err := seg.MidSideEffect(0.5, false); err != nil {
		t.Fatalf("width 0.5 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{0.25, -0.25, -0.125, 0.125})

	// Widening to 2x doubles the side. The raw output {1, -1, ...} hits the
	// peak exactly, so no peak-protection scaling is needed.
	seg = makeStereoSegment([2]float32{0.5, -0.5}, [2]float32{-0.25, 0.25})
	if err := seg.MidSideEffect(2.0, false); err != nil {
		t.Fatalf("width 2 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{1.0, -1.0, -0.5, 0.5})
	if peak := seg.GetMaxAbsValue(); peak > 1.0 {
		t.Fatalf("peak protection failed: peak is %v", peak)
	}
}

// Single-side (left-only) pulses contain equal amounts of mid and side, so
// every width produces a predictable, energy-checked result.
func TestMidSideSingleSidePulses(t *testing.T) {
	newSeg := func() *Segment {
		return makeStereoSegment([2]float32{1.0, 0.0}, [2]float32{-1.0, 0.0})
	}
	original := newSeg()
	originalEnergy := energy(original.Data)

	// Identity.
	seg := newSeg()
	if err := seg.MidSideEffect(1.0, false); err != nil {
		t.Fatalf("default width returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, original.Data)
	if e := energy(seg.Data); math.Abs(e-originalEnergy) > midsideEpsilon {
		t.Fatalf("identity must preserve energy: got %v, want %v", e, originalEnergy)
	}

	// Mono fold-down: mid = +/- 0.5 on both channels.
	seg = newSeg()
	if err := seg.MidSideEffect(0.0, false); err != nil {
		t.Fatalf("width 0 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{0.5, 0.5, -0.5, -0.5})

	// Half width.
	seg = newSeg()
	if err := seg.MidSideEffect(0.5, false); err != nil {
		t.Fatalf("width 0.5 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{0.75, 0.25, -0.75, -0.25})

	// Double width peaks at 1.5, so peak protection scales uniformly by 2/3.
	// Raw output would be (1.5, -0.5), (-1.5, 0.5).
	seg = newSeg()
	if err := seg.MidSideEffect(2.0, false); err != nil {
		t.Fatalf("width 2 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{1.0, -1.0 / 3.0, -1.0, 1.0 / 3.0})
	if peak := seg.GetMaxAbsValue(); peak > 1.0+midsideEpsilon {
		t.Fatalf("peak protection failed: peak is %v", peak)
	}

	// Triple width would peak at 2.0: uniform gain of 0.5 brings it back to
	// 1.0 while keeping the width ratio.
	seg = newSeg()
	if err := seg.MidSideEffect(3.0, false); err != nil {
		t.Fatalf("width 3 returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, []float32{1.0, -0.5, -1.0, 0.5})
	if peak := seg.GetMaxAbsValue(); peak > 1.0+midsideEpsilon {
		t.Fatalf("peak protection failed: peak is %v", peak)
	}

	// Energy check against the analytic M/S formula on a sub-peak width.
	// l=1, r=0: mid=0.5, side=0.5*w, so per frame E = (0.5+0.5w)^2 + (0.5-0.5w)^2
	// = 0.5*(1 + w*w); two frames (one negated, same squares).
	seg = newSeg()
	width := 0.5
	if err := seg.MidSideEffect(width, false); err != nil {
		t.Fatalf("width %v returned error: %v", width, err)
	}
	wantEnergy := 2.0 * 0.5 * (1 + width*width)
	if e := energy(seg.Data); math.Abs(e-wantEnergy) > midsideEpsilon {
		t.Fatalf("energy mismatch: got %v, want %v", e, wantEnergy)
	}
}

// Width and mono preview must never change duration, sample rate or channels.
func TestMidSidePreservesStructure(t *testing.T) {
	seg := makeStereoSegment(
		[2]float32{0.3, -0.3},
		[2]float32{0.1, 0.9},
		[2]float32{-0.7, 0.2},
		[2]float32{0.4, 0.4},
	)
	before := seg.Clone()

	for _, width := range []float64{0, 0.25, 1, 2.5, 10} {
		seg = before.Clone()
		if err := seg.MidSideEffect(width, false); err != nil {
			t.Fatalf("width %v returned error: %v", width, err)
		}
		if len(seg.Data) != len(before.Data) {
			t.Fatalf("width %v changed data length: got %d, want %d", width, len(seg.Data), len(before.Data))
		}
		if seg.Channels != before.Channels || seg.SampleRate != before.SampleRate ||
			seg.Length != before.Length || seg.LengthSeconds != before.LengthSeconds ||
			seg.Format != before.Format {
			t.Fatalf("width %v changed segment metadata", width)
		}
	}

	// Mono preview flag is equivalent to width 0.
	preview := before.Clone()
	if err := preview.MidSideEffect(42.0, true); err != nil {
		t.Fatalf("mono preview returned error: %v", err)
	}
	folded := before.Clone()
	if err := folded.MidSideEffect(0.0, false); err != nil {
		t.Fatalf("width 0 returned error: %v", err)
	}
	approxSliceEqual(t, preview.Data, folded.Data)

	if err := before.MonoPreview(); err != nil {
		t.Fatalf("MonoPreview returned error: %v", err)
	}
	approxSliceEqual(t, before.Data, folded.Data)
}

// Mono segments are already mono-compatible and must be left untouched.
func TestMidSideMonoInput(t *testing.T) {
	seg := SegmentFromData([]float32{0.1, -0.2, 0.3, 0.4, -0.5}, 1, 44100)
	before := seg.Clone()

	if err := seg.MidSideEffect(0.5, false); err != nil {
		t.Fatalf("mono input returned error: %v", err)
	}
	if err := seg.MidSideEffect(5.0, true); err != nil {
		t.Fatalf("mono preview on mono input returned error: %v", err)
	}
	approxSliceEqual(t, seg.Data, before.Data)
	if seg.Channels != 1 || seg.SampleRate != 44100 {
		t.Fatal("mono segment metadata was modified")
	}
}

// An odd-length interleaved buffer keeps its trailing, unpaired sample.
func TestMidSideOddSampleCount(t *testing.T) {
	// 5 values: two full stereo frames plus one dangling sample.
	data := []float32{0.6, -0.6, 0.2, 0.8, 0.42}
	seg := SegmentFromData(append([]float32(nil), data...), 2, 22050)

	if err := seg.MidSideEffect(0.0, false); err != nil {
		t.Fatalf("odd-length processing returned error: %v", err)
	}
	if len(seg.Data) != len(data) {
		t.Fatalf("data length changed: got %d, want %d", len(seg.Data), len(data))
	}

	// First frame is anti-phase -> folds to silence; second frame is mono.
	approxSliceEqual(t, seg.Data[:4], []float32{0, 0, 0.5, 0.5})

	// The dangling sample must survive untouched.
	if !approxEqual(seg.Data[4], 0.42) {
		t.Fatalf("trailing sample changed: got %v, want 0.42", seg.Data[4])
	}
}

// Non-finite and negative coefficients are rejected and leave data unchanged.
func TestMidSideInvalidWidth(t *testing.T) {
	invalid := []float64{-0.01, -1, math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, width := range invalid {
		seg := makeStereoSegment([2]float32{0.6, -0.6}, [2]float32{0.2, 0.8})
		before := seg.Clone()

		err := seg.MidSideEffect(width, false)
		if err == nil {
			t.Fatalf("width %v should return an error", width)
		}
		approxSliceEqual(t, seg.Data, before.Data)
	}
}

// All-silence stays silent for every width and never trips peak protection.
func TestMidSideSilence(t *testing.T) {
	for _, channels := range []int{1, 2} {
		seg := Silent(0.1, 44100, uint(channels))
		beforeLen := len(seg.Data)

		for _, width := range []float64{0, 0.5, 1, 100} {
			if err := seg.MidSideEffect(width, false); err != nil {
				t.Fatalf("silent segment width %v returned error: %v", width, err)
			}
			if len(seg.Data) != beforeLen {
				t.Fatalf("silent segment length changed for width %v", width)
			}
			for i, v := range seg.Data {
				if v != 0 {
					t.Fatalf("silence corrupted at sample %d: got %v (width %v, channels %d)", i, v, width, channels)
				}
			}
		}
	}
}

// Peak protection must keep every processed sample within -1..1, even for
// extreme widths on signals that mix mid and side.
func TestMidSidePeakProtection(t *testing.T) {
	seg := makeStereoSegment(
		[2]float32{0.9, -0.8},
		[2]float32{-0.7, 0.6},
		[2]float32{0.5, 0.4},
	)

	if err := seg.MidSideEffect(20.0, false); err != nil {
		t.Fatalf("extreme width returned error: %v", err)
	}
	for i, v := range seg.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite output at sample %d: %v", i, v)
		}
		if v > 1.0+midsideEpsilon || v < -1.0-midsideEpsilon {
			t.Fatalf("sample %d out of range: %v", i, v)
		}
	}
	if peak := seg.GetMaxAbsValue(); peak > 1.0+midsideEpsilon {
		t.Fatalf("peak exceeds allowed range: %v", peak)
	}
	if len(seg.Data) != 6 {
		t.Fatalf("length changed: got %d, want 6", len(seg.Data))
	}
}

// The effect must compose with the existing overlay/export chain: process a
// segment, overlay another one, export to WAV and read the result back, then
// run the echo effect (which itself uses Overlay) on the processed signal.
func TestMidSideOverlayExportChain(t *testing.T) {
	base := makeStereoSegment(
		[2]float32{0.8, -0.8},
		[2]float32{0.4, 0.4},
		[2]float32{-0.3, 0.6},
		[2]float32{0.2, -0.1},
		[2]float32{0.7, 0.2},
		[2]float32{-0.5, -0.4},
		[2]float32{0.1, 0.9},
		[2]float32{-0.6, 0.3},
	)
	beforeLen := len(base.Data)
	overlaySeg := makeStereoSegment(
		[2]float32{-0.5, 0.5},
		[2]float32{0.1, -0.1},
		[2]float32{0.3, 0.2},
	)

	if err := base.MidSideEffect(1.75, false); err != nil {
		t.Fatalf("mid/side processing returned error: %v", err)
	}
	if len(base.Data) != beforeLen {
		t.Fatalf("mid/side changed length: got %d, want %d", len(base.Data), beforeLen)
	}
	if err := base.Overlay(overlaySeg, 0); err != nil {
		t.Fatalf("overlay returned error: %v", err)
	}
	if len(base.Data) != beforeLen {
		t.Fatalf("overlay changed length: got %d, want %d", len(base.Data), beforeLen)
	}

	for i, v := range base.Data {
		if v > 1.0+midsideEpsilon || v < -1.0-midsideEpsilon {
			t.Fatalf("post-overlay sample %d out of range: %v", i, v)
		}
	}

	var buf bytes.Buffer
	if err := base.Export(&buf, FormatWAVE); err != nil {
		t.Fatalf("export returned error: %v", err)
	}

	reader, err := wav.New(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("wav decode returned error: %v", err)
	}
	if int(reader.SampleRate) != int(base.SampleRate) || int(reader.NumChannels) != base.Channels {
		t.Fatalf("exported header mismatch: rate %d/%d, channels %d/%d",
			reader.SampleRate, base.SampleRate, reader.NumChannels, base.Channels)
	}

	// NOTE: the pre-existing Export passes len(Data)*2 as the WAV frame
	// count, so the header declares twice the bytes actually written. Read
	// only the interleaved samples contained in the data chunk itself.
	decoded, err := reader.ReadFloats(len(base.Data))
	if err != nil {
		t.Fatalf("wav read returned error: %v", err)
	}
	if len(decoded) != len(base.Data) {
		t.Fatalf("round-trip length mismatch: got %d, want %d", len(decoded), len(base.Data))
	}
	// go-dsp decodes int16 with an unsigned-style mapping centered at 0.5
	// ((v - MinInt16) / 65535), so convert back to signed -1..1 before
	// comparing. The signal content must survive the export chain within
	// int16 quantization error.
	for i, want := range base.Data {
		got := decoded[i]*2 - 1
		if math.Abs(float64(got-want)) > 1e-3 {
			t.Fatalf("round-trip sample %d mismatch: got %v, want %v", i, got, want)
		}
	}

	// Echo still works on top of the processed segment and keeps the length.
	echoSeg := base.Clone()
	echoSeg.EchoEffect(1.0/float64(base.SampleRate), 2)
	if len(echoSeg.Data) != beforeLen {
		t.Fatalf("echo changed length: got %d, want %d", len(echoSeg.Data), beforeLen)
	}
	// A one-frame delay means the echo's second layer adds into frame 1, so
	// the chain demonstrably mixed signals instead of being a no-op.
	if approxSliceEqualSilent(echoSeg.Data[2:4], base.Data[2:4]) {
		t.Fatal("echo overlay had no effect on the processed segment")
	}
}

func approxSliceEqualSilent(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}
