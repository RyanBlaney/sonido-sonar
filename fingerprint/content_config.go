package fingerprint

import (
	"github.com/RyanBlaney/sonido-sonar/fingerprint/analyzers"
	"github.com/RyanBlaney/sonido-sonar/fingerprint/config"
)

// ContentSettings holds all content-specific configuration
type ContentSettings struct {
	FeatureSettings    FeatureSettings    `json:"feature_settings"`
	ComparisonSettings ComparisonSettings `json:"comparison_settings"`
}

// FeatureSettings holds feature-specific configuration
type FeatureSettings struct {
	EnableMFCC             bool                 `json:"enable_mfcc"`
	EnableChroma           bool                 `json:"enable_chroma"`
	EnableSpectralContrast bool                 `json:"enable_spectral_contrast"`
	EnableHarmonicFeatures bool                 `json:"enable_harmonic_features"`
	EnableSpeechFeatures   bool                 `json:"enable_speech_features"`
	EnableTemporalFeatures bool                 `json:"enable_temporal_features"`
	SimilarityWeights      map[string]float64   `json:"similarity_weights"`
	MFCCCoefficients       int                  `json:"mfcc_coefficients"`
	ChromaBins             int                  `json:"chroma_bins"`
	WindowType             analyzers.WindowType `json:"window_type"`
}

// ComparisonSettings holds comparison-specific configuration
type ComparisonSettings struct {
	SimilarityThreshold float64            `json:"similarity_threshold"`
	FeatureWeights      map[string]float64 `json:"feature_weights"`
	ToleranceFactors    map[string]float64 `json:"tolerance_factors"`
}

// ContentAwareConfigManager centralizes all content-aware configuration
type ContentAwareConfigManager struct {
	baseConfig     *FingerprintConfig
	contentConfigs map[config.ContentType]ContentSettings
}

// NewContentAwareConfigManager creates a new content-aware config manager
func NewContentAwareConfigManager(baseConfig *FingerprintConfig) *ContentAwareConfigManager {
	if baseConfig == nil {
		baseConfig = DefaultFingerprintConfig()
	}

	return &ContentAwareConfigManager{
		baseConfig:     baseConfig,
		contentConfigs: getContentConfigs(),
	}
}

// GetGenerationConfig returns a complete fingerprint config optimized for the content type
func (c *ContentAwareConfigManager) GetGenerationConfig(contentType config.ContentType) *FingerprintConfig {
	// Start with base config
	generationConfig := *c.baseConfig // Copy

	// Get content-specific settings
	contentSettings, exists := c.contentConfigs[contentType]
	if !exists {
		// Use default/unknown content settings
		contentSettings = c.contentConfigs[config.ContentUnknown]
	}

	// Apply content-specific feature configuration
	generationConfig.FeatureConfig = c.buildFeatureConfig(contentSettings.FeatureSettings)

	return &generationConfig
}

// GetComparisonConfig returns comparison configuration optimized for the content type
func (c *ContentAwareConfigManager) GetComparisonConfig(contentType config.ContentType) *ComparisonConfig {
	contentSettings, exists := c.contentConfigs[contentType]
	if !exists {
		contentSettings = c.contentConfigs[config.ContentUnknown]
	}

	return &ComparisonConfig{
		SimilarityThreshold: contentSettings.ComparisonSettings.SimilarityThreshold,
		FeatureWeights:      contentSettings.ComparisonSettings.FeatureWeights,
		ToleranceFactors:    contentSettings.ComparisonSettings.ToleranceFactors,
		ContentType:         contentType,
	}
}

// buildFeatureConfig converts ContentSettings to FeatureConfig
func (c *ContentAwareConfigManager) buildFeatureConfig(settings FeatureSettings) *config.FeatureConfig {
	return &config.FeatureConfig{
		EnableMFCC:             settings.EnableMFCC,
		EnableChroma:           settings.EnableChroma,
		EnableSpectralContrast: settings.EnableSpectralContrast,
		EnableHarmonicFeatures: settings.EnableHarmonicFeatures,
		EnableSpeechFeatures:   settings.EnableSpeechFeatures,
		EnableTemporalFeatures: settings.EnableTemporalFeatures,
		MFCCCoefficients:       settings.MFCCCoefficients,
		ChromaBins:             settings.ChromaBins,
		SimilarityWeights:      settings.SimilarityWeights,
		WindowType:             settings.WindowType,
		// Copy other settings from base config
		WindowSize: c.baseConfig.FeatureConfig.WindowSize,
		HopSize:    c.baseConfig.FeatureConfig.HopSize,
	}
}

// getContentConfigs returns the centralized content-specific configurations
func getContentConfigs() map[config.ContentType]ContentSettings {
	return map[config.ContentType]ContentSettings{
		config.ContentMusic: {
			FeatureSettings: FeatureSettings{
				EnableMFCC:             true,
				EnableChroma:           true,
				EnableSpectralContrast: true,
				EnableHarmonicFeatures: true,
				EnableSpeechFeatures:   false,
				EnableTemporalFeatures: false,
				MFCCCoefficients:       13,
				ChromaBins:             12,
				WindowType:             analyzers.WindowHann,
				SimilarityWeights: map[string]float64{
					"mfcc":     0.35,
					"chroma":   0.30,
					"harmonic": 0.20,
					"spectral": 0.15,
				},
			},
			ComparisonSettings: ComparisonSettings{
				SimilarityThreshold: 0.75,
				FeatureWeights: map[string]float64{
					"mfcc":     0.35,
					"chroma":   0.30,
					"harmonic": 0.20,
					"spectral": 0.15,
				},
				ToleranceFactors: map[string]float64{
					"pitch":  0.1,  // Music is sensitive to pitch changes
					"tempo":  0.2,  // Allow moderate tempo variations
					"timbre": 0.15, // Moderate timbre tolerance
				},
			},
		},

		config.ContentNews: {
			FeatureSettings: FeatureSettings{
				EnableMFCC:             true,
				EnableChroma:           false,
				EnableSpectralContrast: true,
				EnableHarmonicFeatures: false,
				EnableSpeechFeatures:   true,
				EnableTemporalFeatures: true,
				MFCCCoefficients:       13,
				ChromaBins:             12,
				WindowType:             analyzers.WindowHann,
				SimilarityWeights: map[string]float64{
					"mfcc":     0.50,
					"speech":   0.25,
					"spectral": 0.15,
					"temporal": 0.10,
				},
			},
			ComparisonSettings: ComparisonSettings{
				SimilarityThreshold: 0.80,
				FeatureWeights: map[string]float64{
					"mfcc":     0.50,
					"speech":   0.25,
					"spectral": 0.15,
					"temporal": 0.10,
				},
				ToleranceFactors: map[string]float64{
					"voice":   0.12, // Moderate tolerance for voice variations
					"pace":    0.25, // Allow pace variations in speech
					"clarity": 0.08, // News should be clear
				},
			},
		},

		config.ContentTalk: {
			FeatureSettings: FeatureSettings{
				EnableMFCC:             true,
				EnableChroma:           false,
				EnableSpectralContrast: true,
				EnableHarmonicFeatures: false,
				EnableSpeechFeatures:   true,
				EnableTemporalFeatures: true,
				MFCCCoefficients:       13,
				ChromaBins:             12,
				WindowType:             analyzers.WindowHann,
				SimilarityWeights: map[string]float64{
					"mfcc":     0.45,
					"speech":   0.30,
					"spectral": 0.15,
					"temporal": 0.10,
				},
			},
			ComparisonSettings: ComparisonSettings{
				SimilarityThreshold: 0.78,
				FeatureWeights: map[string]float64{
					"mfcc":     0.30,
					"spectral": 0.25,
					"temporal": 0.25,
					"energy":   0.20,
				},
				ToleranceFactors: map[string]float64{
					"crowd":      0.35, // High tolerance for crowd noise variations
					"commentary": 0.20, // Moderate tolerance for commentary
					"action":     0.25, // Allow for action intensity variations
				},
			},
		},

		config.ContentMixed: {
			FeatureSettings: FeatureSettings{
				EnableMFCC:             true,
				EnableChroma:           true,
				EnableSpectralContrast: true,
				EnableHarmonicFeatures: true,
				EnableSpeechFeatures:   true,
				EnableTemporalFeatures: true,
				MFCCCoefficients:       13,
				ChromaBins:             12,
				WindowType:             analyzers.WindowHann,
				SimilarityWeights: map[string]float64{
					"mfcc":     0.30,
					"spectral": 0.20,
					"temporal": 0.20,
					"chroma":   0.15,
					"speech":   0.15,
				},
			},
			ComparisonSettings: ComparisonSettings{
				SimilarityThreshold: 0.72,
				FeatureWeights: map[string]float64{
					"mfcc":     0.30,
					"spectral": 0.20,
					"temporal": 0.20,
					"chroma":   0.15,
					"speech":   0.15,
				},
				ToleranceFactors: map[string]float64{
					"variation": 0.25, // High tolerance for mixed content
					"segments":  0.30, // Allow for different content segments
					"balance":   0.20, // Account for balance changes
				},
			},
		},

		config.ContentUnknown: {
			FeatureSettings: FeatureSettings{
				EnableMFCC:             true,
				EnableChroma:           true,
				EnableSpectralContrast: true,
				EnableHarmonicFeatures: false, // Conservative for performance
				EnableSpeechFeatures:   false, // Conservative for performance
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
			ComparisonSettings: ComparisonSettings{
				SimilarityThreshold: 0.75, // Balanced threshold
				FeatureWeights: map[string]float64{
					"mfcc":     0.40,
					"spectral": 0.25,
					"chroma":   0.20,
					"temporal": 0.15,
				},
				ToleranceFactors: map[string]float64{
					"general": 0.20, // Moderate tolerance for unknown content
				},
			},
		},
	}
}

// ComparisonConfig represents configuration for fingerprint comparison
type ComparisonConfig struct {
	SimilarityThreshold float64            `json:"similarity_threshold"`
	FeatureWeights      map[string]float64 `json:"feature_weights"`
	ToleranceFactors    map[string]float64 `json:"tolerance_factors"`
	ContentType         config.ContentType `json:"content_type"`
}
