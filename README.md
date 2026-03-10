
# Sonido-Sonar 🎵

A pure Go audio fingerprinting and temporal alignment library. Originally developed during my Summer 2025 internship at
**TuneIn Inc.** to benchmark CDN stream end-to-end latency relative to their source streams.

Calculations are performed based on content type. To optimize performance, you can pass in the content type to
skip the acoustic content type detection phase.

[![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)]()

---

## 📖 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Fingerprinting](#basic-fingerprinting)
  - [Temporal Alignment](#temporal-alignment-example)
  - [Stream Comparison](#stream-comparison-example)
- [Modules & Structure](#modules--structure)
- [Content Detection](#content-detection)
- [Configuration](#configuration)
- [Technical Reference](#technical-reference)
- [License](#license)

---

## ✨ Features

| Category | Capabilities |
|----------|--------------|
| **Fingerprinting** | Multi-feature audio signatures (MFCC, spectral, chroma, speech features) |
| **Alignment** | Temporal alignment using DTW and cross-correlation (up to configurable lag offset) |
| **Content Detection** | Auto-detect music, speech, news, sports, and mixed content types |
| **Spectral Analysis** | MFCC, spectral contrast/centroid/flux, Mel/Bark scales, STFT/FFT |
| **Windowing** | 9 window functions: Hamming, Hann, Blackman, Kaiser, Welch & more |
| **Comparison** | Similarity scoring tailored by content type (music vs speech optimized weights) |
| **CDN Latency** | End-to-end latency measurement for streaming pipeline diagnostics |

---

## 📦 Installation

```bash
go get github.com/RyanBlaney/sonido-sonar
```

Or add to your `go.mod`:

```bash
require github.com/RyanBlaney/sonido-sonar v0.1.0
```

---

## 🚀 Quick Start

### Basic Fingerprint Generation

```go
package main

import (
    "fmt"
    "github.com/RyanBlaney/sonido-sonar/fingerprint"
    "github.com/RyanBlaney/sonido-sonar/transcode"
)

func main() {
    // Load audio data via the decoder
    config := &transcode.DecoderConfig{
        TargetSampleRate: 44100,
        EnableNormalization: true,
    }
    
    decoder := transcode.NewDecoder(config)
    audioData, err := decoder.DecodeFile("path/to/audio.wav")
    if err != nil {
        panic(err)
    }

    // Generate fingerprint
    fpConfig := &fingerprint.Config{
        WindowSize:  1024,
        HopSize:     256,
        ContentType: "music", // or "speech", "news", "sports"
    }
    
    generator := fingerprint.NewFingerprintGenerator(fpConfig)
    fp, err := generator.GenerateFingerprint(audioData)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Generated fingerprint with %d feature points\n", len(fp.Features))
}
```

### Temporal Alignment (Two Streams)

```go
package main

import (
    "fmt"
    "github.com/RyanBlaney/sonido-sonar/fingerprint"
    "github.com/RyanBlaney/sonido-sonar/fingerprint/extractors"
)

func main() {
    // Assume fp1 and fp2 are fingerprints from source & CDN streams
    fp1 := loadFingerprint("source.mp3")
    fp2 := loadFingerprint("cdn_stream.mp3")

    extractor := extractors.NewAlignmentExtractorWithMaxLag(
        fpConfig,
        alignmentConfig,
        maxOffsetSeconds: 60.0,
    )

    features, err := extractor.ExtractAlignmentFeatures(fp1.Features, fp2.Features, 
        srcPCM, cdnPCM, sampleRate)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Detected latency: %.3f seconds\n", features.TemporalOffset)
    fmt.Printf("Alignment confidence: %.2f%%\n", features.OffsetConfidence*100)
}
```

### Stream Similarity Comparison

```go
package main

import (
    "github.com/RyanBlaney/sonido-sonar/fingerprint"
)

func main() {
    fpConfig := fingerprint.Config{ContentType: "music"}
    
    // Generate fingerprints for both streams
    f1 := generateFingerprintForStream(sourceURL, fpConfig)
    f2 := generateFingerprintForStream(cdnURL, fpConfig)

    comparator := fingerprint.NewFingerprintComparator(&fpConfig.ComparisonSettings)
    result, err := comparator.Compare(f1, f2)

    fmt.Printf("Overall Similarity: %.2f%%\n", result.OverallSimilarity*100)
    fmt.Printf("Confidence Score:   %.2f%%\n", result.Confidence*100)
}
```

---

## 📁 Modules & Structure

```
├── algorithms/      # DSP and feature extraction primitives
│   ├── chroma/     # Tonal features (HPCP, pitch class, key detection)
│   ├── spectral/   # STFT, MFCC, power spectrum, centroid/rolloff/etc.
│   ├── speech/     # LPC, voice quality metrics for spoken content
│   ├── rhythm/     # Tempo & meter estimation
│   ├── windowing/  # Window functions (Hamming, Hann, Kaiser...)
│   ├── stats/      # Statistical tools: DTW, correlation, clustering
├── fingerprint/    # Core fingerprint module
│   ├── extractors/* # Content-specific feature extractors (music/speech)
│   └── analyzers/*  # Fingerprint comparison logic
├── transcode/     # Audio decoding & format normalization
└── logging/       # Structured logging interfaces
```

---

## 🎯 Content Detection

The library automatically detects content type to optimize feature extraction:

| Content Type | Optimal Features | Use Case Example |
|--------------|------------------|------------------|
| `music`      | Chroma + Spectral + MFCC | Music streaming services |
| `speech`     | LPC, Pitch Detection, ZCR | Podcasts, Talk Radio |
| `news`       | Speech Features + Energy | News Broadcasts |
| `sports`     | Mixed (energy spikes, speech) | Live Sports Commentary |

```go
detector := fingerprint.NewContentDetector(&fingerprint.Config{AutoDetect: true})
contentType := detector.DetectContentType(audioData) // Returns "music"|"speech"|...
```

---

## ⚙ Configuration

Key configuration parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WindowSize`     | 1024    | FFT window size |
| `HopSize`        | 256     | Frame hop (for STFT) |
| `ContentType`    | "music" | Content type for weighting |
| `EnableSpeechFeatures` | false  | Enable LPC & speech analysis |
| `MFCCCoefficients` | 13      | Number of MFCC to extract |
| `MinSimilarity` | 0.70    | Acceptable fingerprint match threshold |

Example config:

```go
config := &fingerprint.Config{
    WindowSize:   1024,
    HopSize:      256,
    ContentType:  "news",
    EnableSpeechFeatures: true,
    MFCCCoefficients: 12,
}
```

---

## 📚 Technical Reference

### Algorithms Implemented

- **MFCC** (Itakura 1975) — mel-frequency cepstral coefficients
- **STFT / DFT** (Cooley & Tukey 1965) — time-frequency analysis
- **Spectral Metrics** — centroid, contrast, flux, rolloff (Brown 1991)
- **Chroma Features** — pitch class profiles & Tonalnetz
- **DTW Alignment** (Dynamic Time Warping)
- **Correlation Analysis** — Pearson / cross-correlation
- **Window Functions** — Hamming, Hann, Kaiser, Welch, Bartlett...

### Key Citations

> Davis, S. B., & Mermelstein, P. (1980). *Comparison of parametric representation...*  
> Haitsma, J., & Kalker, T. (2002). *A highly robust audio fingerprinting system.*

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for more info.

---

```
Built with ❤️ at TuneIn Inc. | Summer 2025 Internship Project
Author: Ryan Blaney
Repo: github.com/RyanBlaney/sonido-sonar
```
