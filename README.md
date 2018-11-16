# HOGDescriptor

Get [HOG Descriptor](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) from gray scale image.

## Usage

```swift
let width, height: Int
let imageBuffer: [Double] // contains gray scale, row major pixel values

let hogDescriptor = HOGDescriptor(orientations: 9,
                                  pixelsPerCell: (8, 8),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l2Hys)

// Raveled N-D array of shape [NumBlocksY, NumBlocksX, cellsPerBlock.y, cellsPerBlock.x, Orientations].
let features: [Double] = hogDescriptor.getDescriptor(data: imageBuffer,
                                                     width: width, 
                                                     height: height)
```

## Compatible with scikit-image
The output is equivalent to [scikit-image](https://github.com/scikit-image/scikit-image)'s `skimage.feature.hog`.

Check [test.py](https://github.com/t-ae/hog-feature-extractor/blob/master/test.py) and [SKImageCompatibilityTests.swift](https://github.com/t-ae/hog-descriptor/blob/master/Tests/HOGDescriptorTests/SKImageCompatibilityTests.swift) what args you should pass to.

## License

[The MIT License](https://github.com/t-ae/hog-feature-extractor/blob/master/LICENSE)