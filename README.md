# HOGFeatureExtractor

Extract HOG feature vector from gray scale image.

## Usage

```swift
let width, height: Int
let imageBuffer: [Double] // contains gray scale, row major pixel values

let extractor = HOGFeatureExtractor(pixelsPerCell: pixelsPerCell,
                                    cellsPerBlock: cellsPerBlock,
                                    orientation: orientations,
                                    normalization: normalization)

let features = extractor.extract(data: imageBuffer,
                                 width: width, 
                                 height: height)
```

## Compatible with scikit-image
The output is equivalent to [scikit-image](https://github.com/scikit-image/scikit-image)'s `skimage.feature.hog`.

Check [test.py](https://github.com/t-ae/hog-feature-extractor/blob/master/test.py) and [SKImageCompatibilityTests.swift](SKImageCompatibilityTests.swift) what args you should pass to.

## License

[The MIT License](https://github.com/t-ae/hog-feature-extractor/blob/master/LICENSE)