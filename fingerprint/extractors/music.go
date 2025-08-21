package extractors

import (
	"errors"
	"fmt"
	"math"

	"github.com/RyanBlaney/sonido-sonar/algorithms/chroma"
	"github.com/RyanBlaney/sonido-sonar/algorithms/common"
	"github.com/RyanBlaney/sonido-sonar/algorithms/filters"
	"github.com/RyanBlaney/sonido-sonar/algorithms/harmonic"
	"github.com/RyanBlaney/sonido-sonar/algorithms/spectral"
	"github.com/RyanBlaney/sonido-sonar/algorithms/temporal"
	"github.com/RyanBlaney/sonido-sonar/algorithms/tonal"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/analyzers"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/config"
	"github.com/RyanBlaney/sonido-sonar/logging"
)

// MusicFeatureExtractor specializes in extracting features relevant to music content
// Focuses on harmonic content, tonal characteristics, rhythmic patterns, and musical structure
type MusicFeatureExtractor struct {
	config *config.FeatureConfig
	logger logging.Logger

	// Music-specific algorithms - harmonic and tonal analysis
	chromaSTFT       *chroma.ChromaSTFT
	chromaCQT        *chroma.ChromaCQT
	hpcp             *chroma.HPCP
	keyEstimator     *tonal.KeyEstimator
	chordDetector    *tonal.ChordDetector
	pitchDetector    *tonal.PitchDetector
	harmonicRatio    *tonal.HarmonicRatioAnalyzer
	inharmonicity    *tonal.InharmonicityAnalyzer
	harmonicTracking *harmonic.HarmonicTracking
	harmonicProduct  *harmonic.HarmonicProduct
	fundamentalEst   *harmonic.FundamentalEstimation

	// Core spectral analysis algorithms
	spectralCentroid  *spectral.SpectralCentroid
	spectralRolloff   *spectral.SpectralRolloff
	spectralBandwidth *spectral.SpectralBandwidth
	spectralFlatness  *spectral.SpectralFlatness
	spectralCrest     *spectral.SpectralCrest
	spectralSlope     *spectral.SpectralSlope
	spectralFlux      *spectral.SpectralFlux
	spectralContrast  *spectral.SpectralContrast
	zeroCrossing      *spectral.ZeroCrossingRate
	mfcc              *spectral.MFCC

	// Temporal analysis algorithms - rhythm and dynamics
	energy           *temporal.Energy
	envelope         *temporal.Envelope
	onsetDetection   *temporal.OnsetDetection
	tempoEstimation  *temporal.TempoEstimation
	attackDecay      *temporal.AttackDecay
	dynamicRange     *temporal.DynamicRange
	silenceDetection *temporal.SilenceDetection

	// Audio preprocessing
	preEmphasis     *filters.PreEmphasis
	dcRemoval       *filters.DCRemoval
	windowGenerator *analyzers.WindowGenerator

	// Feature weights for music content
	featureWeights map[string]float64
}

// NewMusicFeatureExtractor creates a new music feature extractor
func NewMusicFeatureExtractor(config *config.FeatureConfig) *MusicFeatureExtractor {
	logger := logging.WithFields(logging.Fields{
		"component":    "music_feature_extractor",
		"content_type": "music",
	})

	extractor := &MusicFeatureExtractor{
		config: config,
		logger: logger,
	}

	extractor.initializeAlgorithms()
	extractor.setupFeatureWeights()

	return extractor
}

func (m *MusicFeatureExtractor) initializeAlgorithms() {
	sampleRate := m.config.SampleRate

	// Initialize harmonic and tonal analysis algorithms
	m.chromaSTFT = chroma.NewChromaSTFTDefault(sampleRate)
	m.chromaCQT = chroma.NewChromaCQTDefault(sampleRate)
	m.hpcp = chroma.NewHPCP(sampleRate)
	m.keyEstimator = tonal.NewKeyEstimator(sampleRate)
	m.chordDetector = tonal.NewChordDetector(sampleRate)
	m.pitchDetector = tonal.NewPitchDetector(sampleRate)
	m.harmonicRatio = tonal.NewHarmonicRatioAnalyzer(sampleRate)
	m.inharmonicity = tonal.NewInharmonicityAnalyzer(sampleRate)
	m.harmonicTracking = harmonic.NewHarmonicTracking(sampleRate, m.config.HopSize)
	m.harmonicProduct = harmonic.NewHarmonicProduct(sampleRate, 10, 80.0, 2000.0)
	m.fundamentalEst = harmonic.NewFundamentalEstimation(sampleRate, 80.0, 2000.0)

	// Initialize core spectral analysis algorithms
	m.spectralCentroid = spectral.NewSpectralCentroid(sampleRate)
	m.spectralRolloff = spectral.NewSpectralRolloff(sampleRate)
	m.spectralBandwidth = spectral.NewSpectralBandwidth(sampleRate)
	m.spectralFlatness = spectral.NewSpectralFlatness()
	m.spectralCrest = spectral.NewSpectralCrest()
	m.spectralSlope = spectral.NewSpectralSlope(sampleRate)
	m.spectralFlux = spectral.NewSpectralFlux()
	m.spectralContrast = spectral.NewSpectralContrast(sampleRate, 6) // 6 bands for music
	m.zeroCrossing = spectral.NewZeroCrossingRate(sampleRate)

	// MFCC with music-optimized parameters
	mfccParams := spectral.MFCCParams{
		NumCoefficients: 13,
		NumMelFilters:   26,
		LowFreq:         0,
		HighFreq:        float64(sampleRate) / 2,
		UseLiftering:    true,
		LifterCoeff:     22,
	}
	m.mfcc = spectral.NewMFCCWithParams(sampleRate, mfccParams)

	// Initialize temporal analysis algorithms
	frameSize := m.config.WindowSize
	hopSize := m.config.HopSize
	m.energy = temporal.NewEnergy(frameSize, hopSize, sampleRate)
	m.envelope = temporal.NewEnvelope()
	m.onsetDetection = temporal.NewOnsetDetection()
	m.tempoEstimation = temporal.NewTempoEstimation()
	m.attackDecay = temporal.NewAttackDecay()
	m.dynamicRange = temporal.NewDynamicRange()
	m.silenceDetection = temporal.NewSilenceDetection()

	// Initialize audio preprocessing
	m.preEmphasis = filters.NewPreEmphasisForContent("music", sampleRate)
	m.dcRemoval = filters.NewDCRemoval()
	m.windowGenerator = analyzers.NewWindowGenerator()

	m.logger.Debug("Initialized all music feature extraction algorithms")
}

func (m *MusicFeatureExtractor) setupFeatureWeights() {
	// Music-specific feature weights - emphasize harmonic and tonal content
	m.featureWeights = map[string]float64{
		// Harmonic features - highest priority for music
		"chroma_features": 1.0,
		"pitch_estimate":  0.9,
		"harmonic_ratio":  0.9,
		"key_detection":   0.8,
		"chord_detection": 0.8,
		"inharmonicity":   0.7,

		// Spectral features - important for timbre
		"spectral_centroid":  0.8,
		"spectral_rolloff":   0.7,
		"spectral_bandwidth": 0.7,
		"spectral_flatness":  0.6,
		"spectral_contrast":  0.8,
		"mfcc":               0.7,

		// Temporal features - important for rhythm and dynamics
		"onset_detection":  0.8,
		"tempo_estimation": 0.8,
		"attack_decay":     0.7,
		"dynamic_range":    0.7,
		"rms_energy":       0.6,

		// Lower priority for music
		"zero_crossing_rate": 0.4,
		"silence_ratio":      0.3,
		"speech_features":    0.1, // Minimal for music
	}
}

// ExtractFeatures extracts music-specific features from audio
func (m *MusicFeatureExtractor) ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error) {
	if spectrogram == nil || len(pcm) == 0 {
		return nil, errors.New("invalid input data")
	}

	logger := m.logger.WithFields(logging.Fields{
		"function":    "ExtractFeatures",
		"pcm_length":  len(pcm),
		"time_frames": spectrogram.TimeFrames,
		"freq_bins":   spectrogram.FreqBins,
	})

	logger.Debug("Starting music feature extraction")

	// Preprocess audio
	processedPCM := m.preprocessAudio(pcm)

	features := &ExtractedFeatures{
		ExtractionMetadata: map[string]any{
			"extractor_type": "music",
			"sample_rate":    sampleRate,
			"duration":       float64(len(pcm)) / float64(sampleRate),
		},
	}

	var err error

	// Extract core spectral features
	if features.SpectralFeatures, err = m.extractSpectralFeatures(spectrogram); err != nil {
		logger.Error(err, "Failed to extract spectral features")
		return nil, fmt.Errorf("spectral feature extraction failed: %w", err)
	}

	// Extract MFCC features (important for music similarity)
	if features.MFCC, err = m.extractMFCCFeatures(spectrogram); err != nil {
		logger.Error(err, "Failed to extract MFCC features")
		return nil, fmt.Errorf("MFCC feature extraction failed: %w", err)
	}

	// Extract chroma features (crucial for music)
	if features.ChromaFeatures, err = m.extractChromaFeatures(spectrogram, processedPCM); err != nil {
		logger.Error(err, "Failed to extract chroma features")
		return nil, fmt.Errorf("chroma feature extraction failed: %w", err)
	}

	// Extract temporal features
	if features.TemporalFeatures, err = m.extractTemporalFeatures(processedPCM, spectrogram); err != nil {
		logger.Error(err, "Failed to extract temporal features")
		return nil, fmt.Errorf("temporal feature extraction failed: %w", err)
	}

	// Extract energy features
	if features.EnergyFeatures, err = m.extractEnergyFeatures(processedPCM, spectrogram); err != nil {
		logger.Error(err, "Failed to extract energy features")
		return nil, fmt.Errorf("energy feature extraction failed: %w", err)
	}

	// Extract harmonic features (crucial for music)
	if features.HarmonicFeatures, err = m.extractHarmonicFeatures(processedPCM, spectrogram); err != nil {
		logger.Error(err, "Failed to extract harmonic features")
		return nil, fmt.Errorf("harmonic feature extraction failed: %w", err)
	}

	logger.Debug("Completed music feature extraction")
	return features, nil
}

func (m *MusicFeatureExtractor) preprocessAudio(pcm []float64) []float64 {
	// Apply DC removal
	dcRemoved := make([]float64, len(pcm))
	for i, sample := range pcm {
		dcRemoved[i] = m.dcRemoval.Process(sample)
	}

	// Apply pre-emphasis optimized for music
	preEmphasized := make([]float64, len(dcRemoved))
	for i, sample := range dcRemoved {
		preEmphasized[i] = m.preEmphasis.Process(sample)
	}

	return preEmphasized
}

func (m *MusicFeatureExtractor) extractSpectralFeatures(spectrogram *analyzers.SpectrogramResult) (*SpectralFeatures, error) {
	features := &SpectralFeatures{}
	numFrames := spectrogram.TimeFrames

	// Initialize slices
	features.SpectralCentroid = make([]float64, numFrames)
	features.SpectralRolloff = make([]float64, numFrames)
	features.SpectralBandwidth = make([]float64, numFrames)
	features.SpectralFlatness = make([]float64, numFrames)
	features.SpectralCrest = make([]float64, numFrames)
	features.SpectralSlope = make([]float64, numFrames)
	features.SpectralFlux = make([]float64, numFrames)
	features.ZeroCrossingRate = make([]float64, numFrames)
	features.SpectralContrast = make([][]float64, numFrames)

	// Extract features frame by frame
	for frame := range numFrames {
		magnitude := spectrogram.Magnitude[frame]

		// Basic spectral features
		centroid := m.spectralCentroid.Compute(magnitude)
		features.SpectralCentroid[frame] = centroid
		features.SpectralRolloff[frame] = m.spectralRolloff.Compute(magnitude, 0.85)
		features.SpectralBandwidth[frame] = m.spectralBandwidth.Compute(magnitude, centroid)
		features.SpectralFlatness[frame] = m.spectralFlatness.Compute(magnitude)
		features.SpectralCrest[frame] = m.spectralCrest.Compute(magnitude)
		features.SpectralSlope[frame] = m.spectralSlope.Compute(magnitude)

		// Spectral flux (needs previous frame)
		if frame > 0 {
			flux := m.spectralFlux.Compute([][]float64{spectrogram.Magnitude[frame-1], magnitude})
			if len(flux) > 0 {
				features.SpectralFlux[frame] = flux[0]
			}
		}

		// Spectral contrast
		features.SpectralContrast[frame] = m.spectralContrast.Compute(magnitude)
	}

	return features, nil
}

func (m *MusicFeatureExtractor) extractMFCCFeatures(spectrogram *analyzers.SpectrogramResult) ([][]float64, error) {
	numFrames := spectrogram.TimeFrames
	mfccFeatures := make([][]float64, numFrames)

	for frame := range numFrames {
		magnitude := spectrogram.Magnitude[frame]

		// Convert magnitude to power spectrum
		powerSpectrum := make([]float64, len(magnitude))
		for i, mag := range magnitude {
			powerSpectrum[i] = mag * mag
		}

		result, err := m.mfcc.Compute(powerSpectrum)
		if err != nil {
			return nil, fmt.Errorf("MFCC computation failed at frame %d: %w", frame, err)
		}
		mfccFeatures[frame] = result.MFCC
	}

	return mfccFeatures, nil
}

func (m *MusicFeatureExtractor) extractChromaFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64) ([][]float64, error) {
	numFrames := spectrogram.TimeFrames
	chromaFeatures := make([][]float64, numFrames)

	frameSize := len(pcm) / numFrames
	hopSize := m.config.HopSize

	// Create window once outside the loop for efficiency
	windowConfig := &analyzers.WindowConfig{
		Type:      analyzers.WindowHann,
		Size:      frameSize,
		Normalize: true,
		Symmetric: true,
	}

	window, err := m.windowGenerator.Generate(windowConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create window: %w", err)
	}

	for frame := range numFrames {
		start := frame * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		frameData := pcm[start:end]
		if len(frameData) < frameSize {
			// Pad with zeros if necessary
			padded := make([]float64, frameSize)
			copy(padded, frameData)
			frameData = padded
		}

		// Use ChromaSTFT for robust chroma extraction
		chromaResult, err := m.chromaSTFT.ComputeChroma(frameData, frameSize, hopSize, window)
		if err != nil {
			return nil, fmt.Errorf("chroma computation failed at frame %d: %w", frame, err)
		}

		// ChromaResult is [][]float64, so take the first (and likely only) frame
		if len(chromaResult) > 0 {
			chromaFeatures[frame] = chromaResult[0]
		} else {
			// Fallback: create empty chroma vector
			chromaFeatures[frame] = make([]float64, 12)
		}
	}

	return chromaFeatures, nil
}

func (m *MusicFeatureExtractor) extractTemporalFeatures(pcm []float64, spectrogram *analyzers.SpectrogramResult) (*TemporalFeatures, error) {
	features := &TemporalFeatures{}

	features.RMSEnergy = m.energy.ComputeShortTimeEnergy(pcm)

	numFrames := len(features.RMSEnergy)
	frameSize := len(pcm) / numFrames

	features.EnvelopeShape = m.envelope.ComputeRMS(pcm, frameSize, m.config.HopSize)

	// Peak and average amplitude
	features.PeakAmplitude = 0
	features.AverageAmplitude = 0
	for _, sample := range pcm {
		absSample := math.Abs(sample)
		if absSample > features.PeakAmplitude {
			features.PeakAmplitude = absSample
		}
		features.AverageAmplitude += absSample
	}
	features.AverageAmplitude /= float64(len(pcm))

	// Dynamic range
	noiseFloor := 10.0
	clipCeiling := 90.0
	features.DynamicRange = m.dynamicRange.ComputeRange(pcm, noiseFloor, clipCeiling)

	// Onset detection for rhythm analysis
	threshold := 0.3
	minInterval := 0.05 // (50ms) to ensure we aren't detecting the same event (note or beat)
	// TODO: make this genre and feature aware!
	// High spectral flux variance → shorter intervals (complex music)
	// Low spectral centroid → longer intervals (bass-heavy content)
	// High zero-crossing rate → shorter intervals (percussive content)
	onsetResult, err := m.onsetDetection.DetectOnsets(pcm, spectrogram.SampleRate, threshold, minInterval)
	if err != nil {
		return nil, err
	}
	features.OnsetDensity = float64(len(onsetResult)) / (float64(len(pcm)) / float64(m.config.SampleRate))

	// Attack times (simplified)
	if len(onsetResult) > 0 {
		features.AttackTime = make([]float64, len(onsetResult))
		for i := range onsetResult {
			features.AttackTime[i] = 0.01 // TODO: need more sophisticated analysis
		}
	}

	// Crest factor per frame
	features.CrestFactor = make([]float64, numFrames)

	for i := range numFrames {
		start := i * frameSize
		end := start + frameSize
		end = min(end, len(pcm))

		frameData := pcm[start:end]
		peak := 0.0
		for _, sample := range frameData {
			if math.Abs(sample) > peak {
				peak = math.Abs(sample)
			}
		}

		if features.RMSEnergy[i] > 0 {
			features.CrestFactor[i] = peak / features.RMSEnergy[i]
		}
	}

	// Silence detection
	silenceThreshold := -40.0
	features.SilenceRatio = m.silenceDetection.ComputeSilenceRatio(pcm, spectrogram.SampleRate, silenceThreshold)

	// Activity level (inverse of silence, normalized)
	features.ActivityLevel = make([]float64, numFrames)
	for i := range features.ActivityLevel {
		features.ActivityLevel[i] = 1.0 - features.SilenceRatio
	}

	return features, nil
}

func (m *MusicFeatureExtractor) extractEnergyFeatures(pcm []float64, spectrogram *analyzers.SpectrogramResult) (*EnergyFeatures, error) {
	features := &EnergyFeatures{}

	// Short-time energy
	features.ShortTimeEnergy = m.energy.ComputeShortTimeEnergy(pcm)

	// Energy variance
	features.EnergyVariance = common.Variance(features.ShortTimeEnergy)

	// Energy entropy per frame
	features.EnergyEntropy = make([]float64, len(features.ShortTimeEnergy))
	for i, energy := range features.ShortTimeEnergy {
		if energy > 0 {
			features.EnergyEntropy[i] = -energy * math.Log2(energy)
		}
	}

	// Loudness range (simplified dynamic range)
	maxEnergy := 0.0
	minEnergy := math.Inf(1)
	for _, energy := range features.ShortTimeEnergy {
		if energy > maxEnergy {
			maxEnergy = energy
		}
		if energy < minEnergy && energy > 0 {
			minEnergy = energy
		}
	}
	if minEnergy != math.Inf(1) && minEnergy > 0 {
		features.LoudnessRange = 20 * math.Log10(maxEnergy/minEnergy)
	}

	// Spectral energy distribution
	numFrames := spectrogram.TimeFrames
	features.LowEnergyRatio = make([]float64, numFrames)
	features.HighEnergyRatio = make([]float64, numFrames)

	for frame := range numFrames {
		magnitude := spectrogram.Magnitude[frame]
		totalEnergy := 0.0
		lowEnergy := 0.0
		highEnergy := 0.0

		// Calculate energy in different frequency bands
		numBins := len(magnitude)
		lowBinCutoff := numBins / 4      // Lower quarter
		highBinCutoff := 3 * numBins / 4 // Upper quarter

		for i, mag := range magnitude {
			energy := mag * mag
			totalEnergy += energy

			if i < lowBinCutoff {
				lowEnergy += energy
			} else if i > highBinCutoff {
				highEnergy += energy
			}
		}

		if totalEnergy > 0 {
			features.LowEnergyRatio[frame] = lowEnergy / totalEnergy
			features.HighEnergyRatio[frame] = highEnergy / totalEnergy
		}
	}

	return features, nil
}

func (m *MusicFeatureExtractor) extractHarmonicFeatures(pcm []float64, spectrogram *analyzers.SpectrogramResult) (*HarmonicFeatures, error) {
	features := &HarmonicFeatures{}
	numFrames := spectrogram.TimeFrames
	frameSize := len(pcm) / numFrames

	// Initialize slices
	features.PitchEstimate = make([]float64, numFrames)
	features.PitchConfidence = make([]float64, numFrames)
	features.VoicingStrength = make([]float64, numFrames)
	features.HarmonicRatio = make([]float64, numFrames)
	features.InharmonicityRatio = make([]float64, numFrames)
	features.TonalCentroid = make([]float64, numFrames)

	// Process frame by frame
	for frame := range numFrames {
		start := frame * frameSize
		end := start + frameSize
		end = min(end, len(pcm))

		frameData := pcm[start:end]
		if len(frameData) < frameSize {
			// Pad with zeros if necessary
			padded := make([]float64, frameSize)
			copy(padded, frameData)
			frameData = padded
		}

		// Pitch detection
		pitchResult, err := m.pitchDetector.DetectPitch(frameData)
		if err != nil {
			// Continue with default values on error
			features.PitchEstimate[frame] = 0
			features.PitchConfidence[frame] = 0
			features.VoicingStrength[frame] = 0
		} else {
			features.PitchEstimate[frame] = pitchResult.Pitch
			features.PitchConfidence[frame] = pitchResult.Confidence
			features.VoicingStrength[frame] = pitchResult.Voicing
		}

		// Harmonic ratio analysis
		harmonicResult, err := m.harmonicRatio.AnalyzeFrame(frameData)
		if err != nil {
			features.HarmonicRatio[frame] = 0
		} else {
			features.HarmonicRatio[frame] = harmonicResult.HarmonicRatio
		}

		// Inharmonicity analysis (if pitch is detected
		if features.PitchEstimate[frame] > 0 && features.PitchConfidence[frame] > 0.5 {
			inharmonicResult, err := m.inharmonicity.AnalyzeFrame(frameData)
			if err != nil {
				features.InharmonicityRatio[frame] = 0
			} else {
				features.InharmonicityRatio[frame] = inharmonicResult.Inharmonicity
			}
		}

		// Tonal centroid (simplified - using spectral centroid weighted by harmonicity)
		spectralCentroid := m.spectralCentroid.Compute(spectrogram.Magnitude[frame])
		features.TonalCentroid[frame] = spectralCentroid * features.VoicingStrength[frame]
	}

	return features, nil
}

// GetFeatureWeights returns the feature weights for music content
func (m *MusicFeatureExtractor) GetFeatureWeights() map[string]float64 {
	return m.featureWeights
}

// GetName returns the name of this extractor
func (m *MusicFeatureExtractor) GetName() string {
	return "music_feature_extractor"
}

// GetContentType returns the content type this extractor is designed for
func (m *MusicFeatureExtractor) GetContentType() config.ContentType {
	return config.ContentMusic
}
