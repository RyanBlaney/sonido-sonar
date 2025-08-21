package fingerprint

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/RyanBlaney/sonido-sonar/fingerprint/extractors"
	"github.com/RyanBlaney/sonido-sonar/transcode"
)

func calculateDuration(audioData *transcode.AudioData) time.Duration {
	if audioData.SampleRate <= 0 {
		return 0
	}
	seconds := float64(len(audioData.PCM)) / float64(audioData.SampleRate*audioData.Channels)
	return time.Duration(seconds * float64(time.Second))
}

func generateID(audioData *transcode.AudioData) string {
	hasher := sha256.New()
	fmt.Fprintf(hasher, "%d_%d_%d",
		time.Now().UnixNano(),
		len(audioData.PCM),
		audioData.SampleRate)
	return hex.EncodeToString(hasher.Sum(nil))[:16]
}

func addMetadata(fingerprint *AudioFingerprint, audioData *transcode.AudioData, extractor extractors.FeatureExtractor, config *FingerprintConfig) {
	fingerprint.Metadata["extractor_name"] = extractor.GetName()
	fingerprint.Metadata["feature_weights"] = extractor.GetFeatureWeights()
	fingerprint.Metadata["generation_time"] = time.Now()
	fingerprint.Metadata["config"] = config

	if audioData.Metadata != nil {
		fingerprint.Metadata["stream_metadata"] = audioData.Metadata
	}

	// Add feature statistics
	if fingerprint.Features != nil {
		stats := make(map[string]any)

		if fingerprint.Features.MFCC != nil {
			stats["mfcc_frames"] = len(fingerprint.Features.MFCC)
			if len(fingerprint.Features.MFCC) > 0 {
				stats["mfcc_coefficients"] = len(fingerprint.Features.MFCC[0])
			}
		}

		if fingerprint.Features.SpectralFeatures != nil {
			stats["spectral_frames"] = len(fingerprint.Features.SpectralFeatures.SpectralCentroid)
		}

		fingerprint.Metadata["feature_stats"] = stats
	}
}
