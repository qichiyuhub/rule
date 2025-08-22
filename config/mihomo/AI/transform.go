package lightgbm

import (
    "fmt"
    "os"
    "strconv"
    "strings"
    "sync"

    "github.com/metacubex/mihomo/log"
)

// parseTransformsContent parses the [transforms] section from the model file.
// The expected format is as follows:
//
// [transforms]
// [order]
// 0=success
// 1=failure
// ... (index=feature_name, one per line, up to MaxFeatureSize)
// [/order]
//
// [definitions]
// std_type=StandardScaler
// std_features=2,3,4,5,6,7,15
// std_mean=...comma separated float values...
// std_scale=...comma separated float values...
//
// robust_type=RobustScaler
// robust_features=0,1
// robust_center=...comma separated float values...
// robust_scale=...comma separated float values...
// [/definitions]
//
// untransformed_features=8:is_udp,9:is_tcp,10:asn_feature,...
// transform=true
// [/transforms]
//
// - Each transform block (e.g. std_*, robust_*) must include:
//   - *_type: the transform type, e.g. "StandardScaler" or "RobustScaler"
//   - *_features: comma-separated feature indices (int, based on order)
//   - other parameters: comma-separated float values, length must match features
// - All indices and parameter arrays must match the feature order and count.
// - Only features listed in *_features are transformed; others remain unchanged.
// - The Go parser expects strict adherence to this structure for correct parsing.

type TransformType string

const (
    StandardScalerTransform TransformType = "StandardScaler"
    RobustScalerTransform   TransformType = "RobustScaler"
)

type TransformParams struct {
    Type           TransformType            `json:"type"`
    FeatureIndices []int                   `json:"feature_indices"`
    Parameters     map[string][]float64    `json:"parameters"`
}

type FeatureTransforms struct {
    TransformsEnabled      bool                        `json:"transforms_enabled"`
    FeatureOrder           map[int]string              `json:"order"`
    Transforms             []TransformParams           `json:"transforms"`
    UntransformedFeatures  []string                    `json:"untransformed_features"`
}

var transformPool = sync.Pool{
    New: func() interface{} {
        return make([]float64, MaxFeatureSize)
    },
}

// 读取transforms参数
func LoadTransformsFromModel(modelPath string) (*FeatureTransforms, error) {
    file, err := os.Open(modelPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open model file: %v", err)
    }
    defer file.Close()

    stat, err := file.Stat()
    if err != nil {
        return nil, fmt.Errorf("failed to get file info: %v", err)
    }

    readSize := int64(16384)
    if stat.Size() < readSize {
        readSize = stat.Size()
    }

    _, err = file.Seek(-readSize, 2)
    if err != nil {
        return nil, fmt.Errorf("failed to seek file position: %v", err)
    }

    buffer := make([]byte, readSize)
    _, err = file.Read(buffer)
    if err != nil {
        return nil, fmt.Errorf("failed to read file content: %v", err)
    }

    content := string(buffer)

    startMarker := "[transforms]"
    endMarker := "[/transforms]"

    startIdx := strings.Index(content, startMarker)
    if startIdx == -1 {
        return &FeatureTransforms{
            TransformsEnabled: false,
            FeatureOrder:      getDefaultFeatureOrder(),
            Transforms:        []TransformParams{},
        }, nil
    }

    endIdx := strings.Index(content, endMarker)
    if endIdx == -1 {
        return nil, fmt.Errorf("found transforms start marker but no end marker")
    }

    transformsContent := content[startIdx+len(startMarker):endIdx]

    featureTransforms, err := parseTransformsContent(transformsContent)
    if err != nil {
        return nil, fmt.Errorf("failed to parse transforms parameters: %v", err)
    }

    return featureTransforms, nil
}

func parseTransformsContent(content string) (*FeatureTransforms, error) {
    featureTransforms := &FeatureTransforms{
        FeatureOrder: make(map[int]string),
        Transforms:   []TransformParams{},
    }

    lines := strings.Split(content, "\n")
    currentSection := ""

    transformDefs := make(map[string]map[string]string)
    errors := []string{}

    for lineNum, line := range lines {
        line = strings.TrimSpace(line)

        if line == "" || strings.HasPrefix(line, "#") {
            continue
        }

        if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
            sectionName := strings.Trim(line, "[]")

            if strings.HasPrefix(sectionName, "/") {
                currentSection = ""
                continue
            } else {
                currentSection = sectionName
                continue
            }
        }

        if !strings.Contains(line, "=") {
            continue
        }

        parts := strings.SplitN(line, "=", 2)
        if len(parts) != 2 {
            continue
        }

        key := strings.TrimSpace(parts[0])
        value := strings.TrimSpace(parts[1])

        switch currentSection {
        case "order":
            idx, err := strconv.Atoi(key)
            if err != nil {
                errors = append(errors, fmt.Sprintf("invalid feature index '%s' at line %d", key, lineNum+1))
                continue
            }
            if idx < 0 || idx >= MaxFeatureSize {
                errors = append(errors, fmt.Sprintf("feature index %d out of range [0, %d) at line %d", idx, MaxFeatureSize, lineNum+1))
                continue
            }
            featureTransforms.FeatureOrder[idx] = value

        case "definitions":
            if strings.Contains(key, "_") {
                parts := strings.SplitN(key, "_", 2)
                if len(parts) == 2 {
                    transformID := parts[0]
                    paramName := parts[1]

                    if transformDefs[transformID] == nil {
                        transformDefs[transformID] = make(map[string]string)
                    }
                    transformDefs[transformID][paramName] = value
                }
            }

        default:
            switch key {
            case "transform":
                featureTransforms.TransformsEnabled = value == "true"
            case "untransformed_features":
                featureTransforms.UntransformedFeatures = parseStringArray(value)
            }
        }
    }

    validTransformCount := 0
    for transformID, params := range transformDefs {
        transform, err := buildTransformParams(params)
        if err != nil {
            errors = append(errors, fmt.Sprintf("failed to build transform %s: %v", transformID, err))
            continue
        }

        if len(transform.FeatureIndices) == 0 {
            errors = append(errors, fmt.Sprintf("transform %s has no feature indices", transformID))
            continue
        }

        validIndices := true
        for _, idx := range transform.FeatureIndices {
            if idx < 0 || idx >= MaxFeatureSize {
                errors = append(errors, fmt.Sprintf("transform %s contains invalid feature index %d", transformID, idx))
                validIndices = false
                break
            }
        }

        if !validIndices {
            continue
        }

        featureTransforms.Transforms = append(featureTransforms.Transforms, *transform)
        validTransformCount++
    }

    if len(featureTransforms.FeatureOrder) == 0 {
        featureTransforms.FeatureOrder = getDefaultFeatureOrder()
    } else {
        expectedCount := 21
        if len(featureTransforms.FeatureOrder) != expectedCount {
            defaultOrder := getDefaultFeatureOrder()
            for idx, name := range defaultOrder {
                if _, exists := featureTransforms.FeatureOrder[idx]; !exists {
                    featureTransforms.FeatureOrder[idx] = name
                }
            }
        }
    }

    if len(errors) > 0 {
        log.Errorln("[Smart] Transform parsing errors: %s", strings.Join(errors, "; "))
    }

    return featureTransforms, nil
}

func buildTransformParams(params map[string]string) (*TransformParams, error) {
    transform := &TransformParams{
        Parameters: make(map[string][]float64),
    }

    if typeStr, exists := params["type"]; exists {
        transform.Type = TransformType(typeStr)
    } else {
        return nil, fmt.Errorf("missing transform type")
    }

    if featuresStr, exists := params["features"]; exists {
        indices, err := parseIntArray(featuresStr)
        if err != nil {
            return nil, fmt.Errorf("failed to parse feature indices: %v", err)
        }
        transform.FeatureIndices = indices
    } else {
        return nil, fmt.Errorf("missing feature indices")
    }

    for paramName, paramValue := range params {
        if paramName != "type" && paramName != "features" {
            values, err := parseFloatArray(paramValue)
            if err != nil {
                return nil, fmt.Errorf("failed to parse parameter %s: %v", paramName, err)
            }
            transform.Parameters[paramName] = values
        }
    }

    return transform, nil
}

func parseFloatArray(value string) ([]float64, error) {
    if value == "" {
        return []float64{}, nil
    }

    parts := strings.Split(value, ",")
    result := make([]float64, len(parts))

    for i, part := range parts {
        val, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
        if err != nil {
            return nil, fmt.Errorf("failed to parse float '%s': %v", part, err)
        }
        result[i] = val
    }

    return result, nil
}

func parseIntArray(value string) ([]int, error) {
    if value == "" {
        return []int{}, nil
    }

    parts := strings.Split(value, ",")
    result := make([]int, len(parts))

    for i, part := range parts {
        val, err := strconv.Atoi(strings.TrimSpace(part))
        if err != nil {
            return nil, fmt.Errorf("failed to parse integer '%s': %v", part, err)
        }
        result[i] = val
    }

    return result, nil
}

func parseStringArray(value string) []string {
    if value == "" {
        return []string{}
    }

    parts := strings.Split(value, ",")
    result := make([]string, len(parts))

    for i, part := range parts {
        result[i] = strings.TrimSpace(part)
    }

    return result
}

func getDefaultFeatureOrder() map[int]string {
    return map[int]string{
        0: "success", 1: "failure", 2: "connect_time", 3: "latency",
        4: "upload_mb", 5: "download_mb", 6: "duration_minutes",
        7: "last_used_seconds", 8: "is_udp", 9: "is_tcp",
        10: "asn_feature", 11: "country_feature", 12: "address_feature",
        13: "port_feature", 14: "traffic_ratio", 15: "traffic_density",
        16: "connection_type_feature", 17: "asn_hash", 18: "host_hash",
        19: "ip_hash", 20: "geoip_hash",
    }
}

func (ft *FeatureTransforms) ApplyTransforms(features []float64) []float64 {
    if ft == nil || !ft.TransformsEnabled || len(ft.Transforms) == 0 {
        return features
    }

    var result []float64
    poolObj := transformPool.Get()
    if arr, ok := poolObj.([]float64); ok && len(arr) >= len(features) {
        result = arr[:len(features)]
    } else {
        result = make([]float64, len(features))
    }
    copy(result, features)

    errors := []string{}
    for i, transform := range ft.Transforms {
        validTransform := true
        for _, idx := range transform.FeatureIndices {
            if idx < 0 || idx >= len(result) {
                errors = append(errors, fmt.Sprintf("transform %d feature index %d out of range", i, idx))
                validTransform = false
                break
            }
        }

        if validTransform {
            ft.applyTransformInPlace(result, transform)
        }
    }

    if len(errors) > 0 {
        log.Errorln("[Smart] Apply transforms errors: %s", strings.Join(errors, "; "))
    }

    transformPool.Put(result)

    out := make([]float64, len(result))
    copy(out, result)
    return out
}

func (ft *FeatureTransforms) applyTransformInPlace(features []float64, transform TransformParams) {
    switch transform.Type {
    case StandardScalerTransform:
        ft.applyStandardScaler(features, transform)
    case RobustScalerTransform:
        ft.applyRobustScaler(features, transform)
    default:
        log.Errorln("[Smart] Unknown transform type: %s", transform.Type)
    }
}

// 标准化
func (ft *FeatureTransforms) applyStandardScaler(features []float64, transform TransformParams) {
    mean := transform.Parameters["mean"]
    scale := transform.Parameters["scale"]

    if len(mean) == 0 || len(scale) == 0 {
        return
    }

    expectedCount := len(transform.FeatureIndices)
    if len(mean) != expectedCount || len(scale) != expectedCount {
        log.Errorln("[Smart] StandardScaler parameter count mismatch, expected %d, got mean=%d scale=%d",
            expectedCount, len(mean), len(scale))
        return
    }

    errors := []string{}
    for i, featureIdx := range transform.FeatureIndices {
        if featureIdx < len(features) && i < len(mean) && i < len(scale) {
            if scale[i] != 0 {
                features[featureIdx] = (features[featureIdx] - mean[i]) / scale[i]
            } else {
                errors = append(errors, fmt.Sprintf("scale[%d] is zero for feature %d", i, featureIdx))
            }
        }
    }

    if len(errors) > 0 {
        log.Errorln("[Smart] StandardScaler errors: %s", strings.Join(errors, "; "))
    }
}

// 鲁棒缩放
func (ft *FeatureTransforms) applyRobustScaler(features []float64, transform TransformParams) {
    center := transform.Parameters["center"]
    scale := transform.Parameters["scale"]

    if len(center) == 0 || len(scale) == 0 {
        return
    }

    for i, featureIdx := range transform.FeatureIndices {
        if featureIdx < len(features) && i < len(center) && i < len(scale) {
            if scale[i] != 0 {
                features[featureIdx] = (features[featureIdx] - center[i]) / scale[i]
            }
        }
    }
}

func (ft *FeatureTransforms) ValidateTransforms(expectedFeatureCount int) error {
    if ft == nil {
        return fmt.Errorf("FeatureTransforms is nil")
    }

    if !ft.TransformsEnabled {
        return nil
    }

    if len(ft.FeatureOrder) == 0 {
        return fmt.Errorf("feature order mapping is empty")
    }

    for i := 0; i < expectedFeatureCount; i++ {
        if _, exists := ft.FeatureOrder[i]; !exists {
            return fmt.Errorf("feature index %d missing in feature order mapping", i)
        }
    }

    for i, transform := range ft.Transforms {
        switch transform.Type {
        case StandardScalerTransform, RobustScalerTransform:
        default:
            return fmt.Errorf("transform %d: unsupported transform type %s", i, transform.Type)
        }

        if len(transform.FeatureIndices) == 0 {
            return fmt.Errorf("transform %d: feature indices list is empty", i)
        }

        for _, idx := range transform.FeatureIndices {
            if idx < 0 || idx >= expectedFeatureCount {
                return fmt.Errorf("transform %d: feature index %d out of range [0, %d)",
                    i, idx, expectedFeatureCount)
            }
        }

        if err := ft.validateTransformParams(transform); err != nil {
            return fmt.Errorf("transform %d parameter validation failed: %v", i, err)
        }
    }

    return nil
}

func (ft *FeatureTransforms) validateTransformParams(transform TransformParams) error {
    switch transform.Type {
    case StandardScalerTransform:
        mean := transform.Parameters["mean"]
        scale := transform.Parameters["scale"]
        if len(mean) != len(transform.FeatureIndices) {
            return fmt.Errorf("StandardScaler mean parameter count mismatch")
        }
        if len(scale) != len(transform.FeatureIndices) {
            return fmt.Errorf("StandardScaler scale parameter count mismatch")
        }
        for i, s := range scale {
            if s == 0 {
                return fmt.Errorf("StandardScaler scale[%d] is zero", i)
            }
        }

    case RobustScalerTransform:
        center := transform.Parameters["center"]
        scale := transform.Parameters["scale"]
        if len(center) != len(transform.FeatureIndices) {
            return fmt.Errorf("RobustScaler center parameter count mismatch")
        }
        if len(scale) != len(transform.FeatureIndices) {
            return fmt.Errorf("RobustScaler scale parameter count mismatch")
        }
        for i, s := range scale {
            if s == 0 {
                return fmt.Errorf("RobustScaler scale[%d] is zero", i)
            }
        }
    }

    return nil
}

func (ft *FeatureTransforms) DebugTransforms() {
    if ft == nil {
        log.Debugln("[Smart] FeatureTransforms is nil")
        return
    }

    transformSummary := make([]string, len(ft.Transforms))
    for i, transform := range ft.Transforms {
        transformSummary[i] = fmt.Sprintf("%s[%d features]", transform.Type, len(transform.FeatureIndices))
    }

    log.Debugln("[Smart] FeatureTransforms: enabled=%v, features=%d, transforms=%d [%s], untransformed=%v",
        ft.TransformsEnabled, len(ft.FeatureOrder), len(ft.Transforms),
        strings.Join(transformSummary, ", "), ft.UntransformedFeatures)
}