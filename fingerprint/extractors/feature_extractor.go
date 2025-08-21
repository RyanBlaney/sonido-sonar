package extractors

import (
	"github.com/RyanBlaney/sonido-sonar/fingerprint/analyzers"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/config"
	"github.com/RyanBlaney/sonido-sonar/logging"
)

// FeatureExtractor defines the interface for content-specific feature extraction
type FeatureExtractor interface {
	ExtractFeatures(spectrogram *analyzers.SpectrogramResult, pcm []float64, sampleRate int) (*ExtractedFeatures, error)
	GetFeatureWeights() map[string]float64
	GetName() string
	GetContentType() config.ContentType
}

// FeatureExtractorFactory creates feature extractors based on content type
type FeatureExtractorFactory struct {
	logger logging.Logger
}

// NewFeatureExtractorFactory creates a new feature extractor factory
func NewFeatureExtractorFactory() *FeatureExtractorFactory {
	return &FeatureExtractorFactory{
		logger: logging.WithFields(logging.Fields{
			"component": "feature_extractor_factory",
		}),
	}
}

// CreateExtractor creates a feature extractor for the specified content type
func (f *FeatureExtractorFactory) CreateExtractor(contentType config.ContentType, featureConfig config.FeatureConfig) (FeatureExtractor, error) {
	logger := f.logger.WithFields(logging.Fields{
		"function":     "CreateExtractor",
		"content_type": contentType,
	})

	switch contentType {
	case config.ContentMusic:
		logger.Info("Creating music feature extractor")
		return NewMusicFeatureExtractor(&featureConfig), nil

	case config.ContentNews:
		logger.Info("Creating news feature extractor")
		return NewSpeechFeatureExtractor(&featureConfig, true), nil

	case config.ContentTalk:
		logger.Info("Creating talk feature extractor")
		return NewSpeechFeatureExtractor(&featureConfig, false), nil

	// case config.ContentSports:
	// logger.Debug("Creating sports feature extractor")
	// return NewSportsFeatureExtractor(&featureConfig), nil

	// case config.ContentMixed:
	// logger.Debug("Creating mixed content feature extractor")
	// return NewMixedFeatureExtractor(&featureConfig), nil

	default:
		logger.Debug("Creating general feature extractor for unknown content")
		return NewSpeechFeatureExtractor(&featureConfig, true), nil
	}
}
