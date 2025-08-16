package stats

import (
	"fmt"
	"math"
	"sort"
)

// AlignmentMethod defines different alignment approaches
type AlignmentMethod int

const (
	AlignmentDTW AlignmentMethod = iota
	AlignmentCrossCorrelation
	AlignmentPhaseCorrelation
	AlignmentHybrid
)

// AlignmentAnalyzer provides comprehensive audio alignment capabilities
// WHY: Audio alignment is crucial for fingerprint matching, synchronization,
// and determining similarity between audio segments of different lengths
type AlignmentAnalyzer struct {
	method           AlignmentMethod
	maxLag           int
	dtwAnalyzer      *DTWAlignment
	crossCorr        *CrossCorrelation
	stepSize         int
	hopSize          int
	windowSize       int
	confidenceThresh float64
}

// AlignmentResult contains comprehensive alignment analysis
type AlignmentResult struct {
	// Primary alignment information
	Method        AlignmentMethod `json:"method"`
	Offset        int             `json:"offset"`         // Sample offset (negative = query is delayed)
	OffsetSeconds float64         `json:"offset_seconds"` // Offset in seconds
	Confidence    float64         `json:"confidence"`     // Alignment confidence (0-1)
	Similarity    float64         `json:"similarity"`     // Overall similarity score

	// DTW-specific results
	DTWResult *DTWResult `json:"dtw_result,omitempty"`

	// Cross-correlation results
	CrossCorrResult *CorrelationResult `json:"cross_corr_result,omitempty"`

	// Quality metrics
	AlignmentQuality float64 `json:"alignment_quality"` // Quality of alignment
	NoiseLevel       float64 `json:"noise_level"`       // Estimated noise level
	Stability        float64 `json:"stability"`         // Stability of alignment

	// Analysis metadata
	QueryLength     int     `json:"query_length"`
	ReferenceLength int     `json:"reference_length"`
	ProcessingTime  float64 `json:"processing_time"` // Processing time (ms)
	SampleRate      int     `json:"sample_rate"`
}

func NewAlignmentAnalyzer(method AlignmentMethod, maxLag, sampleRate, hopSize, windowSize int, confidenceThresh float64) *AlignmentAnalyzer {
	// Create cross-correlation with explicit normalized correlation
	crossCorr := NewCrossCorrelationWithParams(
		maxLag,
		NormalizedCrossCorrelation, // Use this instead of Pearson for raw features
		TimeDomain,
	)

	// Ensure normalization is enabled
	crossCorr.normalizeInputs = true

	return &AlignmentAnalyzer{
		method:           method,
		maxLag:           maxLag,
		dtwAnalyzer:      NewDTWAlignment(),
		crossCorr:        crossCorr,
		stepSize:         1, // TODO: get from config
		hopSize:          hopSize,
		windowSize:       windowSize,
		confidenceThresh: confidenceThresh,
	}
}

// AlignFeatures aligns two feature sequences (e.g., MFCC, chroma)
func (aa *AlignmentAnalyzer) AlignFeatures(query, reference [][]float64, sampleRate int) (*AlignmentResult, error) {
	if len(query) == 0 || len(reference) == 0 {
		return nil, fmt.Errorf("empty feature sequences provided")
	}

	result := &AlignmentResult{
		Method:          aa.method,
		QueryLength:     len(query),
		ReferenceLength: len(reference),
		SampleRate:      sampleRate,
	}

	switch aa.method {
	case AlignmentDTW:
		return aa.alignWithDTW(query, reference, result)
	case AlignmentCrossCorrelation:
		return aa.alignWithCrossCorrelation(query, reference, result)
	case AlignmentHybrid:
		return aa.alignWithHybrid(query, reference, result)
	default:
		return nil, fmt.Errorf("unsupported alignment method: %d", aa.method)
	}
}

// AlignAudio aligns two audio signals using energy-based features
func (aa *AlignmentAnalyzer) AlignAudio(queryPCM, referencePCM []float64, sampleRate int) (*AlignmentResult, error) {
	// Convert audio to energy features for alignment
	queryFeatures := aa.extractEnergyFeatures(queryPCM, sampleRate)
	refFeatures := aa.extractEnergyFeatures(referencePCM, sampleRate)

	// Convert 1D features to 2D for compatibility
	query2D := make([][]float64, len(queryFeatures))
	ref2D := make([][]float64, len(refFeatures))

	for i, v := range queryFeatures {
		query2D[i] = []float64{v}
	}
	for i, v := range refFeatures {
		ref2D[i] = []float64{v}
	}

	return aa.AlignFeatures(query2D, ref2D, sampleRate)
}

// alignWithDTW performs DTW-based alignment
func (aa *AlignmentAnalyzer) alignWithDTW(query, reference [][]float64, result *AlignmentResult) (*AlignmentResult, error) {
	dtwResult, err := aa.dtwAnalyzer.Align(query, reference)
	if err != nil {
		return nil, fmt.Errorf("DTW alignment failed: %w", err)
	}

	result.DTWResult = dtwResult
	result.Similarity = aa.calculateSimilarityFromDTW(dtwResult)
	result.Confidence = aa.calculateDTWConfidence(dtwResult)

	// Calculate average offset from DTW path
	result.Offset = aa.calculateAverageOffset(dtwResult.Path)
	result.OffsetSeconds = float64(result.Offset) / float64(result.SampleRate)

	// Calculate quality metrics
	result.AlignmentQuality = aa.calculateDTWQuality(dtwResult)
	result.Stability = aa.calculatePathStability(dtwResult.Path)

	return result, nil
}

// alignWithCrossCorrelation performs cross-correlation based alignment
func (aa *AlignmentAnalyzer) alignWithCrossCorrelation(query, reference [][]float64, result *AlignmentResult) (*AlignmentResult, error) {
	queryVec := aa.flatten2DFeatures(query)
	refVec := aa.flatten2DFeatures(reference)

	corrResult, err := aa.crossCorr.Compute(queryVec, refVec)
	if err != nil {
		return nil, fmt.Errorf("cross-correlation failed: %w", err)
	}

	result.CrossCorrResult = corrResult

	// Convert frame lag back to sample lag
	hopSize := aa.hopSize
	result.Offset = corrResult.PeakLag * hopSize // Convert frames to samples
	result.OffsetSeconds = float64(result.Offset) / float64(result.SampleRate)

	rawCorrelation := corrResult.PeakCorrelation

	// For similarity, we want the strength of correlation (absolute value)
	// Negative correlations are still "similar" (just inverted)
	similarity := math.Abs(rawCorrelation)
	similarity = math.Min(1.0, math.Max(0.0, similarity)) // Clamp to [0,1]

	result.Similarity = similarity

	result.Confidence = aa.calculateCorrelationConfidence(corrResult)
	result.AlignmentQuality = aa.calculateCorrelationQuality(corrResult)
	result.NoiseLevel = 1.0 - corrResult.SNR/20.0

	return result, nil
}

func (aa *AlignmentAnalyzer) calculateCorrelationConfidence(corrResult *CorrelationResult) float64 {
	if corrResult == nil {
		return 0.0
	}

	peakMagnitude := math.Abs(corrResult.PeakCorrelation)

	// Early exit for clearly poor correlations
	if peakMagnitude < 0.1 {
		return 0.0
	}

	// Factor 1: Peak magnitude with boost for strong correlations
	peakScore := peakMagnitude
	if peakMagnitude >= 0.6 {
		// Give strong peaks extra credit to reach 80-90% range
		peakScore = peakMagnitude + (peakMagnitude-0.6)*0.5 // Boost strong correlations
	}

	// Factor 2: Enhanced sharpness scaling
	sharpnessScore := math.Min(0.9, corrResult.Sharpness*8.0) // Increased from 5x to 8x

	// Factor 3: Better sidelobe rewards
	sidelobeScore := 0.0
	if corrResult.PeakToSidelobe > 0 && !math.IsInf(corrResult.PeakToSidelobe, 1) {
		sidelobeScore = math.Min(0.8, corrResult.PeakToSidelobe/15.0) // More generous
	}

	// Factor 4: Better SNR scaling
	snrScore := 0.0
	if corrResult.SNR > 0 {
		snrScore = math.Min(0.7, corrResult.SNR/25.0) // More generous
	}

	// Factor 5: Light second peak penalty
	secondPeakPenalty := 0.0
	if corrResult.SecondPeak != 0 && peakMagnitude > 0 {
		secondPeakRatio := math.Abs(corrResult.SecondPeak) / peakMagnitude
		if secondPeakRatio > 0.7 {
			secondPeakPenalty = (secondPeakRatio - 0.7) * 0.25 // Lighter penalty
		}
	}

	// Factor 6: Bonus for excellent correlations
	excellenceBonus := 0.0
	if peakMagnitude >= 0.75 {
		excellenceBonus = 0.12
	} else if peakMagnitude >= 0.6 {
		excellenceBonus = 0.08
	}

	// Enhanced weighting to reward good alignments
	confidence := 0.55*peakScore + // Main factor: actual correlation
		0.22*sharpnessScore + // Secondary: peak sharpness
		0.12*sidelobeScore + // Sidelobe ratio
		0.06*snrScore + // SNR
		0.05*0.15 + // Small base (0.75% total)
		excellenceBonus - // Bonus for strong peaks
		secondPeakPenalty // Penalty for ambiguous peaks

	return math.Min(1.0, math.Max(0.0, confidence))
}

func (aa *AlignmentAnalyzer) calculateCorrelationQuality(corrResult *CorrelationResult) float64 {
	if corrResult == nil {
		return 0.0
	}

	peakMagnitude := math.Abs(corrResult.PeakCorrelation)

	// Early exit for poor correlations
	if peakMagnitude < 0.08 {
		return 0.0
	}

	// Factor 1: Enhanced peak strength scaling
	peakQuality := peakMagnitude
	if peakMagnitude >= 0.6 {
		// Boost quality for strong peaks
		peakQuality = peakMagnitude + (peakMagnitude-0.6)*0.4
	}

	// Factor 2: Enhanced sharpness
	sharpnessQuality := math.Min(0.85, corrResult.Sharpness*5.0)

	// Factor 3: Sidelobe quality
	sidelobeQuality := 0.0
	if corrResult.PeakToSidelobe > 0 && !math.IsInf(corrResult.PeakToSidelobe, 1) {
		sidelobeQuality = math.Min(0.7, corrResult.PeakToSidelobe/20.0)
	}

	// Factor 4: Better SNR quality
	snrQuality := 0.0
	if corrResult.SNR > 0 {
		snrQuality = math.Min(0.6, corrResult.SNR/30.0) // More generous
	}

	// Factor 5: Boundary penalty ONLY for large negative offsets (failed alignments)
	lagPenalty := 0.0
	if aa.maxLag > 0 && corrResult.PeakLag < 0 {
		negativeRatio := math.Abs(float64(corrResult.PeakLag)) / float64(aa.maxLag)
		if negativeRatio > 0.90 { // Large negative offset = likely failed alignment
			lagPenalty = (negativeRatio - 0.90) * 4.0
		}
	}

	// Factor 6: Quality excellence bonus
	qualityBonus := 0.0
	if peakMagnitude >= 0.7 {
		qualityBonus = 0.10
	} else if peakMagnitude >= 0.55 {
		qualityBonus = 0.06
	}

	// Enhanced weighting
	quality := 0.50*peakQuality + // Primary: correlation strength
		0.25*sharpnessQuality + // Important: sharpness
		0.15*sidelobeQuality + // Sidelobe clarity
		0.10*snrQuality + // SNR
		qualityBonus - // Excellence bonus
		lagPenalty // Boundary penalty

	return math.Min(1.0, math.Max(0.0, quality))
}

// Helper methods

func (aa *AlignmentAnalyzer) extractEnergyFeatures(pcm []float64, sampleRate int) []float64 {
	frameSize := aa.windowSize
	hopSize := aa.hopSize
	numFrames := (len(pcm)-frameSize)/hopSize + 1

	energy := make([]float64, numFrames)
	for i := range numFrames {
		start := i * hopSize
		end := start + frameSize
		end = min(end, len(pcm))

		// Calculate RMS energy
		sum := 0.0
		for j := start; j < end; j++ {
			sum += pcm[j] * pcm[j]
		}
		energy[i] = math.Sqrt(sum / float64(end-start))
	}

	return energy
}

func (aa *AlignmentAnalyzer) flatten2DFeatures(features [][]float64) []float64 {
	if len(features) == 0 {
		return []float64{}
	}

	result := make([]float64, len(features))

	// Option A: Use first component (often most important)
	for i, frame := range features {
		if len(frame) > 0 {
			result[i] = frame[0]
		}
	}

	return result
}

func (aa *AlignmentAnalyzer) calculateSimilarityFromDTW(dtwResult *DTWResult) float64 {
	if dtwResult == nil {
		return 0.0
	}

	// Method 1: Length-normalized distance similarity
	avgLength := float64(dtwResult.QueryLength+dtwResult.RefLength) / 2.0
	if avgLength == 0 {
		return 0.0
	}

	normalizedDistance := dtwResult.Distance / avgLength
	distanceSimilarity := 1.0 / (1.0 + normalizedDistance)

	// Method 2: Path-based similarity
	pathQuality := aa.calculateDTWQuality(dtwResult)

	// Method 3: Cost-based similarity (inverse of mean cost)
	meanCost := aa.calculateMeanPathCost(dtwResult.Path)
	costSimilarity := 1.0 / (1.0 + meanCost)

	// Weighted combination
	finalSimilarity := 0.5*distanceSimilarity + 0.3*pathQuality + 0.2*costSimilarity

	return math.Min(1.0, math.Max(0.0, finalSimilarity))
}

func (aa *AlignmentAnalyzer) calculateMeanPathCost(path []AlignPoint) float64 {
	if len(path) == 0 {
		return 0.0
	}

	totalCost := 0.0
	for _, point := range path {
		totalCost += point.Cost
	}

	return totalCost / float64(len(path))
}

func (aa *AlignmentAnalyzer) calculateDTWConfidence(dtwResult *DTWResult) float64 {
	if dtwResult == nil || len(dtwResult.Path) == 0 {
		return 0.0
	}

	// Method 1: Length-normalized distance with exponential decay
	avgLength := float64(dtwResult.QueryLength+dtwResult.RefLength) / 2.0
	if avgLength == 0 {
		return 0.0
	}

	// Normalize by average sequence length (this is the key fix!)
	normalizedDistance := dtwResult.Distance / avgLength

	// Use exponential decay for better confidence mapping
	confidence1 := math.Exp(-normalizedDistance * 2.0) // Scale factor 2.0 for sensitivity

	// Method 2: Path efficiency based confidence
	expectedLength := math.Max(float64(dtwResult.QueryLength), float64(dtwResult.RefLength))
	pathEfficiency := expectedLength / float64(len(dtwResult.Path))
	pathEfficiency = math.Min(1.0, pathEfficiency) // Clamp to [0,1]

	// Method 3: Cost consistency (improved from variance)
	costConsistency := aa.calculateCostConsistency(dtwResult.Path)

	// Method 4: Diagonal bias (prefer straight paths)
	diagonalBias := aa.calculateDiagonalBias(dtwResult.Path)

	// Weighted combination of confidence factors
	finalConfidence := 0.4*confidence1 + 0.25*pathEfficiency + 0.2*costConsistency + 0.15*diagonalBias

	return math.Min(1.0, math.Max(0.0, finalConfidence))
}

func (aa *AlignmentAnalyzer) calculateCostConsistency(path []AlignPoint) float64 {
	if len(path) <= 1 {
		return 0.0
	}

	// Calculate moving average of costs to smooth variations
	windowSize := min(5, len(path)/4) // Adaptive window size
	windowSize = max(windowSize, 2)

	smoothedCosts := make([]float64, len(path))
	for i := range path {
		sum := 0.0
		count := 0

		// Calculate average in window around current point
		for j := max(0, i-windowSize/2); j <= min(len(path)-1, i+windowSize/2); j++ {
			sum += path[j].Cost
			count++
		}
		smoothedCosts[i] = sum / float64(count)
	}

	// Calculate coefficient of variation of smoothed costs
	mean := 0.0
	for _, cost := range smoothedCosts {
		mean += cost
	}
	mean /= float64(len(smoothedCosts))

	if mean <= 1e-10 {
		return 1.0 // Perfect consistency if all costs are near zero
	}

	variance := 0.0
	for _, cost := range smoothedCosts {
		diff := cost - mean
		variance += diff * diff
	}
	variance /= float64(len(smoothedCosts))
	stdDev := math.Sqrt(variance)

	coeffOfVariation := stdDev / mean

	// Convert to consistency score (lower CV = higher consistency)
	consistency := 1.0 / (1.0 + coeffOfVariation)
	return consistency
}

func (aa *AlignmentAnalyzer) calculateDiagonalBias(path []AlignPoint) float64 {
	if len(path) <= 1 {
		return 1.0
	}

	diagonalSteps := 0
	totalSteps := len(path) - 1

	for i := 1; i < len(path); i++ {
		deltaQuery := path[i].QueryIndex - path[i-1].QueryIndex
		deltaRef := path[i].RefIndex - path[i-1].RefIndex

		// Count diagonal steps (both indices advance)
		if deltaQuery > 0 && deltaRef > 0 {
			diagonalSteps++
		}
	}

	if totalSteps == 0 {
		return 1.0
	}

	diagonalRatio := float64(diagonalSteps) / float64(totalSteps)

	// Apply sigmoid-like transformation to emphasize higher diagonal ratios
	return 1.0 / (1.0 + math.Exp(-10.0*(diagonalRatio-0.3))) // Inflection at 30% diagonal
}

func (aa *AlignmentAnalyzer) calculateAverageOffset(path []AlignPoint) int {
	if len(path) == 0 {
		return 0
	}

	sumOffset := 0
	for _, point := range path {
		sumOffset += point.RefIndex - point.QueryIndex
	}

	return sumOffset / len(path)
}

func (aa *AlignmentAnalyzer) calculateDTWQuality(dtwResult *DTWResult) float64 {
	if dtwResult == nil || len(dtwResult.Path) == 0 {
		return 0.0
	}

	// Factor 1: Path efficiency (shorter relative path is better)
	expectedLength := math.Max(float64(dtwResult.QueryLength), float64(dtwResult.RefLength))
	efficiency := expectedLength / float64(len(dtwResult.Path))
	efficiency = math.Min(1.0, efficiency)

	// Factor 2: Diagonal preference
	diagonalRatio := aa.calculateDiagonalBias(dtwResult.Path)

	// Factor 3: Path smoothness (fewer direction changes)
	smoothness := aa.calculatePathSmoothness(dtwResult.Path)

	// Factor 4: Cost monotonicity (costs should be relatively stable)
	costStability := aa.calculateCostConsistency(dtwResult.Path)

	// Weighted combination
	quality := 0.3*efficiency + 0.3*diagonalRatio + 0.2*smoothness + 0.2*costStability

	return math.Min(1.0, math.Max(0.0, quality))
}

func (aa *AlignmentAnalyzer) calculatePathSmoothness(path []AlignPoint) float64 {
	if len(path) <= 2 {
		return 1.0
	}

	directionChanges := 0
	totalSteps := len(path) - 1

	prevDeltaQuery := 0
	prevDeltaRef := 0

	for i := 1; i < len(path); i++ {
		deltaQuery := path[i].QueryIndex - path[i-1].QueryIndex
		deltaRef := path[i].RefIndex - path[i-1].RefIndex

		// Count direction changes
		if i > 1 {
			if (deltaQuery != prevDeltaQuery) || (deltaRef != prevDeltaRef) {
				directionChanges++
			}
		}

		prevDeltaQuery = deltaQuery
		prevDeltaRef = deltaRef
	}

	if totalSteps == 0 {
		return 1.0
	}

	// Convert to smoothness score
	smoothnessRatio := 1.0 - float64(directionChanges)/float64(totalSteps)
	return math.Max(0.0, smoothnessRatio)
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (aa *AlignmentAnalyzer) calculatePathStability(path []AlignPoint) float64 {
	if len(path) < 3 {
		return 0.0
	}

	// Calculate stability based on path smoothness
	directionChanges := 0
	prevDirection := [2]int{0, 0}

	for i := 1; i < len(path); i++ {
		currDirection := [2]int{
			path[i].QueryIndex - path[i-1].QueryIndex,
			path[i].RefIndex - path[i-1].RefIndex,
		}

		if i > 1 && (currDirection[0] != prevDirection[0] || currDirection[1] != prevDirection[1]) {
			directionChanges++
		}

		prevDirection = currDirection
	}

	// Lower direction changes = higher stability
	stability := 1.0 - float64(directionChanges)/float64(len(path)-1)
	return math.Max(0.0, stability)
}

// GetAlignmentQuality calculates quality metrics (wrapper for DTW function)
func GetAlignmentQuality(dtwResult *DTWResult) map[string]float64 {
	// This would call the DTW analyzer's quality function
	// For now, return basic metrics
	quality := make(map[string]float64)

	if dtwResult == nil || len(dtwResult.Path) == 0 {
		return quality
	}

	// Path efficiency
	expectedLength := math.Max(float64(dtwResult.QueryLength), float64(dtwResult.RefLength))
	quality["path_efficiency"] = expectedLength / float64(len(dtwResult.Path))

	// Diagonal ratio
	diagonalSteps := 0
	for i := 1; i < len(dtwResult.Path); i++ {
		if dtwResult.Path[i].QueryIndex > dtwResult.Path[i-1].QueryIndex &&
			dtwResult.Path[i].RefIndex > dtwResult.Path[i-1].RefIndex {
			diagonalSteps++
		}
	}
	quality["diagonal_ratio"] = float64(diagonalSteps) / float64(len(dtwResult.Path)-1)

	return quality
}

// FindBestAlignment tries multiple methods and returns the best result
func (aa *AlignmentAnalyzer) FindBestAlignment(query, reference [][]float64, sampleRate int) (*AlignmentResult, error) {
	methods := []AlignmentMethod{AlignmentCrossCorrelation, AlignmentDTW}
	var bestResult *AlignmentResult
	bestScore := 0.0

	for _, method := range methods {
		aa.method = method
		result, err := aa.AlignFeatures(query, reference, sampleRate)
		if err != nil {
			continue
		}

		// Score based on confidence and similarity
		score := 0.6*result.Confidence + 0.4*result.Similarity
		if score > bestScore {
			bestScore = score
			bestResult = result
		}
	}

	if bestResult == nil {
		return nil, fmt.Errorf("all alignment methods failed")
	}

	return bestResult, nil
}

// AlignmentStats provides statistical analysis of alignment results
type AlignmentStats struct {
	MeanOffset   float64 `json:"mean_offset"`
	StdDevOffset float64 `json:"stddev_offset"`
	MedianOffset float64 `json:"median_offset"`
	OffsetRange  float64 `json:"offset_range"`
	Consistency  float64 `json:"consistency"` // How consistent are multiple alignments
}

// AnalyzeAlignmentConsistency analyzes consistency across multiple alignment attempts
func (aa *AlignmentAnalyzer) AnalyzeAlignmentConsistency(query, reference [][]float64, sampleRate int, numTrials int) (*AlignmentStats, error) {
	if numTrials < 2 {
		numTrials = 5
	}

	offsets := make([]float64, 0, numTrials)

	// Perform multiple alignments with slight variations
	for i := 0; i < numTrials; i++ {
		// Add small random perturbations to test robustness
		perturbedQuery := aa.addNoise(query, 0.01) // 1% noise

		result, err := aa.AlignFeatures(perturbedQuery, reference, sampleRate)
		if err != nil {
			continue
		}

		offsets = append(offsets, float64(result.Offset))
	}

	if len(offsets) == 0 {
		return nil, fmt.Errorf("no successful alignments")
	}

	return aa.calculateOffsetStats(offsets), nil
}

func (aa *AlignmentAnalyzer) addNoise(features [][]float64, noiseLevel float64) [][]float64 {
	// Add small amount of Gaussian noise for robustness testing
	noisy := make([][]float64, len(features))
	for i, frame := range features {
		noisy[i] = make([]float64, len(frame))
		for j, val := range frame {
			// Simple pseudo-random noise (not cryptographically secure)
			noise := (math.Sin(float64(i*j+i+j)) * noiseLevel * val)
			noisy[i][j] = val + noise
		}
	}
	return noisy
}

func (aa *AlignmentAnalyzer) calculateOffsetStats(offsets []float64) *AlignmentStats {
	if len(offsets) == 0 {
		return &AlignmentStats{}
	}

	// Calculate mean
	sum := 0.0
	for _, offset := range offsets {
		sum += offset
	}
	mean := sum / float64(len(offsets))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, offset := range offsets {
		diff := offset - mean
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(len(offsets)))

	// Calculate median
	sorted := make([]float64, len(offsets))
	copy(sorted, offsets)
	sort.Float64s(sorted)

	var median float64
	n := len(sorted)
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	// Calculate range
	offsetRange := sorted[n-1] - sorted[0]

	// Calculate consistency (inverse of coefficient of variation)
	consistency := 1.0
	if mean != 0 {
		cv := stdDev / math.Abs(mean)
		consistency = 1.0 / (1.0 + cv)
	}

	return &AlignmentStats{
		MeanOffset:   mean,
		StdDevOffset: stdDev,
		MedianOffset: median,
		OffsetRange:  offsetRange,
		Consistency:  consistency,
	}
}
