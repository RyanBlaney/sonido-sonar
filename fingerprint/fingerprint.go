package fingerprint

import (
	"fmt"
	"time"

	"github.com/RyanBlaney/sonido-sonar/fingerprint/analyzers"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/config"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/extractors"
	"github.com/RyanBlaney/sonido-sonar/logging"
	"github.com/RyanBlaney/sonido-sonar/transcode"
)

// AudioFingerprint represents a complete audio fingerprint
type AudioFingerprint struct {
	ID          string                        `json:"id"`
	StreamURL   string                        `json:"stream_url"`
	ContentType config.ContentType            `json:"content_type"`
	Timestamp   time.Time                     `json:"timestamp"`
	Duration    time.Duration                 `json:"duration"`
	SampleRate  int                           `json:"sample_rate"`
	HopSize     int                           `json:"hop_size"` // for alignment
	Channels    int                           `json:"channels"`
	Features    *extractors.ExtractedFeatures `json:"features"`
	Metadata    map[string]any                `json:"metadata,omitempty"`
}

// FingerprintConfig holds configuration for fingerprint generation
type FingerprintConfig struct {
	WindowSize          int                        `json:"window_size"`
	HopSize             int                        `json:"hop_size"`
	EnableContentDetect bool                       `json:"enable_content_detect"`
	FeatureConfig       *config.FeatureConfig      `json:"feature_config"`
	ContentConfig       *config.ContentAwareConfig `json:"content_config"`
}

// FingerprintGenerator generates audio fingerprints
type FingerprintGenerator struct {
	config           *FingerprintConfig
	contentManager   *ContentAwareConfigManager
	extractorFactory *extractors.FeatureExtractorFactory
	ContentDetector  *ContentDetector
	spectralAnalyzer *analyzers.SpectralAnalyzer
	logger           logging.Logger
}

// NewFingerprintGenerator creates a new fingerprint generator with configuration
func NewFingerprintGenerator(config *FingerprintConfig) *FingerprintGenerator {
	if config == nil {
		config = DefaultFingerprintConfig()
	}

	logger := logging.WithFields(logging.Fields{
		"component": "fingerprint_generator",
	})

	contentManager := NewContentAwareConfigManager(config)

	return &FingerprintGenerator{
		config:           config,
		contentManager:   contentManager,
		extractorFactory: extractors.NewFeatureExtractorFactory(),
		spectralAnalyzer: analyzers.NewSpectralAnalyzer(44100), // Will be updated with actual sample rate
		ContentDetector:  NewContentDetector(config.ContentConfig),
		logger:           logger,
	}
}

// DefaultFingerprintConfig return default fingerprint configuration
func DefaultFingerprintConfig() *FingerprintConfig {
	return &FingerprintConfig{
		WindowSize:          2048,
		HopSize:             512,
		EnableContentDetect: true,
		FeatureConfig: &config.FeatureConfig{
			EnableMFCC:             true,
			EnableChroma:           true,
			EnableSpectralContrast: true,
			EnableHarmonicFeatures: false, // for performance
			EnableSpeechFeatures:   false, // enabled for speech content
			EnableTemporalFeatures: true,
			MFCCCoefficients:       13,
			ChromaBins:             12,
			WindowType:             analyzers.WindowHann,
			SimilarityWeights: map[string]float64{
				"mfcc":     0.40,
				"spectral": 0.25,
				"chroma":   0.20,
				"temporal": 0.15,
			},
		},
		ContentConfig: &config.ContentAwareConfig{
			EnableContentDetection: true,
			DefaultContentType:     config.ContentUnknown,
			AutoDetectThreshold:    2.0,
		},
	}
}

func ContentOptimizedFingerprintConfig(contentType config.ContentType) *FingerprintConfig {
	cfg := DefaultFingerprintConfig()

	// Adjust other settings based on content type
	switch contentType {
	case config.ContentNews, config.ContentTalk:
		cfg.FeatureConfig.EnableMFCC = true
		cfg.FeatureConfig.EnableSpeechFeatures = true
		cfg.FeatureConfig.EnableSpectralContrast = true
		cfg.FeatureConfig.EnableTemporalFeatures = true
		cfg.FeatureConfig.EnableChroma = false
		cfg.FeatureConfig.EnableHarmonicFeatures = false

	case config.ContentMusic:
		cfg.FeatureConfig.EnableMFCC = true
		cfg.FeatureConfig.EnableChroma = true
		cfg.FeatureConfig.EnableHarmonicFeatures = true
		cfg.FeatureConfig.EnableSpectralContrast = true
		cfg.FeatureConfig.EnableSpeechFeatures = false
		cfg.FeatureConfig.EnableTemporalFeatures = false

	case config.ContentSports:
		cfg.FeatureConfig.EnableMFCC = true
		cfg.FeatureConfig.EnableTemporalFeatures = true
		cfg.FeatureConfig.EnableSpectralContrast = true
		cfg.FeatureConfig.EnableSpeechFeatures = false
		cfg.FeatureConfig.EnableChroma = false
		cfg.FeatureConfig.EnableHarmonicFeatures = false

	default:
		// Use all hash types for unknown content
	}

	return cfg
}

// GenerateFingerprint generates a complete audio fingerprint from audio data
func (fg *FingerprintGenerator) GenerateFingerprint(audioData *transcode.AudioData) (*AudioFingerprint, *config.FeatureConfig, error) {
	if audioData == nil {
		return nil, nil, fmt.Errorf("audio data cannot be nil")
	}

	logger := fg.logger.WithFields(logging.Fields{
		"function":    "GenerateFingerprint",
		"sample_rate": audioData.SampleRate,
		"channels":    audioData.Channels,
		"samples":     len(audioData.PCM),
	})

	logger.Debug("Starting fingerprint generation")

	// Update spectral analyzer with correct sample rate
	fg.spectralAnalyzer = analyzers.NewSpectralAnalyzer(audioData.SampleRate)

	// Detect content type if enabled
	contentType := config.ToContentType(audioData.Metadata.ContentType)
	if contentType == config.ContentUnknown && fg.config.EnableContentDetect {
		contentType = fg.ContentDetector.DetectContentType(audioData)
		logger.Debug("Auto-detected content type", logging.Fields{
			"content_type": contentType,
		})
	} else if contentType != config.ContentUnknown {
		logger.Debug("Using endpoint content type", logging.Fields{
			"content_type": contentType,
		})
	}

	// Update feature config based on content type
	generationConfig := fg.contentManager.GetGenerationConfig(contentType)

	// Extract features using appropriate extractor
	extractor, err := fg.extractorFactory.CreateExtractor(contentType, *fg.config.FeatureConfig)
	if err != nil {
		logger.Error(err, "Failed to create feature extractor")
		return nil, fg.config.FeatureConfig, err
	}

	windowSize := fg.config.WindowSize
	generationConfig.WindowSize = windowSize
	generationConfig.FeatureConfig.WindowSize = windowSize

	hopSize := fg.config.FeatureConfig.HopSize
	generationConfig.HopSize = hopSize
	generationConfig.FeatureConfig.HopSize = hopSize

	windowType := fg.config.FeatureConfig.WindowType
	generationConfig.FeatureConfig.WindowType = windowType

	logger.Debug("STFT Configuration", logging.Fields{
		"window_size": windowSize,
		"hop_size":    hopSize,
		"pcm_length":  len(audioData.PCM),
	})

	spectrogram, err := fg.spectralAnalyzer.ComputeSTFTWithWindow(
		audioData.PCM,
		windowSize,
		hopSize,
		windowType)
	if err != nil {
		logger.Error(err, "Failed to compute STFT")
	}

	logger.Debug("STFT Debug Info", logging.Fields{
		"pcm_length":     len(audioData.PCM),
		"window_size":    windowSize,
		"hop_size":       hopSize,
		"stft_frames":    spectrogram.TimeFrames,
		"stft_freq_bins": spectrogram.FreqBins,
	})

	features, err := extractor.ExtractFeatures(spectrogram, audioData.PCM, audioData.SampleRate)
	if err != nil {
		logger.Error(err, "Failed to extract features")
		return nil, fg.config.FeatureConfig, err
	}

	// Generate fingerprint
	fingerprint := &AudioFingerprint{
		ID:          generateID(audioData),
		StreamURL:   audioData.Metadata.URL,
		ContentType: contentType,
		Timestamp:   time.Now(),
		Duration:    calculateDuration(audioData),
		SampleRate:  audioData.SampleRate,
		HopSize:     fg.config.FeatureConfig.HopSize,
		Channels:    audioData.Channels,
		Features:    features,
		Metadata:    make(map[string]any),
	}

	// Add metadata
	addMetadata(fingerprint, audioData, extractor, fg.config)

	logger.Debug("Fingerprint generation completed", logging.Fields{
		"fingerprint_id": fingerprint.ID,
		"content_type":   fingerprint.ContentType,
	})

	return fingerprint, fg.config.FeatureConfig, nil
}
