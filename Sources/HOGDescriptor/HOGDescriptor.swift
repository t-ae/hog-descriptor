import Foundation
import Accelerate

public class HOGDescriptor {
    
    public enum NormalizationMethod {
        case l1, l1sqrt, l2, l2Hys
    }
    
    public let pixelsPerCell: (x: Int, y: Int)
    public let cellsPerBlock: (x: Int, y: Int)
    public let orientations: Int
    
    public let normalization: NormalizationMethod
    public let transformSqrt: Bool
    
    public let eps = 1e-5
    
    /// Create HOGFeatureExtractor.
    /// - Parameters:
    ///   - orientation: Number of orientation bins. default: 9
    ///   - pixelsPerCell: Size (in pixels) of a cell. default: (8, 8)
    ///   - cellsPerBlock: Number of cells in each block. default: (3, 3)
    ///   - normalization: Block normalization method. default: .l1
    ///   - transformSqrt: Apply power law compression to normalize the image before processing. default: false
    public init(orientations: Int = 9,
                pixelsPerCell: (x: Int, y: Int) = (8, 8),
                cellsPerBlock: (x: Int, y: Int) = (3, 3),
                normalization: NormalizationMethod = .l1,
                transformSqrt: Bool = false) {
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.orientations = orientations
        self.normalization = normalization
        self.transformSqrt = transformSqrt
    }
    
    /// Create HOGFeatureExtractor with square cells/blocks.
    /// - Parameters:
    ///   - orientations: Number of orientation bins. default: 9
    ///   - cellSpan: Size (in pixels) of a cell.
    ///   - blockSpan: Number of cells in each block.
    ///   - normalization: Block normalization method. default: .l1
    ///   - transformSqrt: Apply power law compression to normalize the image before processing. default: false
    public convenience init(orientations: Int = 9,
                            cellSpan: Int,
                            blockSpan: Int,
                            normalization: NormalizationMethod = .l1,
                            transformSqrt: Bool = false) {
        self.init(orientations: orientations,
                  pixelsPerCell: (cellSpan, cellSpan),
                  cellsPerBlock: (blockSpan, blockSpan),
                  normalization: normalization,
                  transformSqrt: transformSqrt)
    }
    
    /// Get size of HOG descriptor for specified width/size image.
    public func getDescriptorSize(width: Int, height: Int) -> Int {
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        let numberOfBlocksX = numberOfCellX - cellsPerBlock.x + 1
        let numberOfBlocksY = numberOfCellY - cellsPerBlock.y + 1
        
        return numberOfBlocksY*numberOfBlocksX*cellsPerBlock.y*cellsPerBlock.x*orientations
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    public func getDescriptor(data: UnsafePointer<Double>,
                              width: Int,
                              height: Int) -> [Double] {
        if transformSqrt {
            var transformed = [Double](repeating: 0, count: width*height)
            var count = Int32(transformed.count)
            vvsqrt(&transformed, data, &count)
            return _getDescriptor(data: transformed, width: width, height: height)
        } else {
            return _getDescriptor(data: data, width: width, height: height)
        }
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    public func getDescriptor(data: UnsafePointer<UInt8>, width: Int, height: Int) -> [Double] {
        var doubleImage = [Double](repeating: 0, count: width*height)
        vDSP_vfltu8D(data, 1, &doubleImage, 1, UInt(doubleImage.count))
        if transformSqrt {
            var count = Int32(doubleImage.count)
            vvsqrt(&doubleImage, doubleImage, &count)
        }
        return _getDescriptor(data: doubleImage, width: width, height: height)
    }
    
    func derivate(data: UnsafePointer<Double>, width: Int, height: Int) -> (x: [Double], y: [Double]) {
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hog.py#L23-L44
        
        var gradX = [Double](repeating: 0, count: width*height)
        gradX.withUnsafeMutableBufferPointer {
            let xImgPtr = $0.baseAddress!
            
            var dpLeft = data
            var dpRight = data.advanced(by: 2)
            var dst = xImgPtr.advanced(by: 1)
            for _ in 0..<height {
                vDSP_vsubD(dpLeft, 1, dpRight, 1, dst, 1, UInt(width-2))
                
                dpLeft += width
                dpRight += width
                dst += width
            }
        }
        
        var gradY = [Double](repeating: 0, count: width*height)
        gradY.withUnsafeMutableBufferPointer {
            let yImgPtr = $0.baseAddress!
            
            let dpUp = data
            let dpDown = data.advanced(by: 2*width)
            let dst = yImgPtr.advanced(by: width)
            
            vDSP_vsubD(dpUp, 1, dpDown, 1, dst, 1, UInt(width*(height-2)))
        }
        
        return (gradX, gradY)
    }
    
    public func _getDescriptor(data: UnsafePointer<Double>,
                               width: Int,
                               height: Int) -> [Double] {
        
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        // derivatives
        let (gradX, gradY) = derivate(data: data, width: width, height: height)
        
        // calculate gradient directions and magnitudes
        var grad = [Double](repeating: 0, count: gradX.count)
        do {
            var _cnt = Int32(grad.count)
            vvatan2(&grad, gradY, gradX, &_cnt) // [-pi, pi]
            var multiplier = Double(orientations) / .pi
            var adder = Double(orientations)
            vDSP_vsmsaD(grad, 1, &multiplier, &adder, &grad, 1, UInt(grad.count)) // [0, 2*orientation]
        }
        
        var magnitude = [Double](repeating: 0, count: gradX.count)
        vDSP_vdistD(gradX, 1, gradY, 1, &magnitude, 1, UInt(magnitude.count))
        
        // accumulate to histograms
        var histograms = [Double](repeating: 0, count: numberOfCellY*numberOfCellX*orientations)
        for y in 0..<height {
            let cellY = y / pixelsPerCell.y
            guard cellY < numberOfCellY else {
                break
            }
            for x in 0..<width {
                let cellX = x / pixelsPerCell.x
                guard cellX < numberOfCellX else {
                    continue
                }
                
                var directionIndex = Int(grad[y*width+x])
                while directionIndex >= orientations {
                    directionIndex -= orientations
                }
                
                let histogramIndex = (cellY * numberOfCellX + cellX) * orientations
                histograms[histogramIndex + directionIndex] += magnitude[y*width+x]
            }
        }
        
        // Scale histograms
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hoghistogram.pyx#L74
        // The final output doesn't differ without this?
        //        var divisor = Double(pixelsPerCell.y * pixelsPerCell.x)
        //        vDSP_vsdivD(histograms, 1, &divisor, &histograms, 1, UInt(histograms.count))
        
        // normalize
        let numberOfBlocksX = numberOfCellX - cellsPerBlock.x + 1
        let numberOfBlocksY = numberOfCellY - cellsPerBlock.y + 1
        
        let featureCount = numberOfBlocksY*numberOfBlocksX*cellsPerBlock.y*cellsPerBlock.x*orientations
        var normalizedHistogram = [Double](repeating: 0, count: featureCount)
        
        normalizedHistogram.withUnsafeMutableBufferPointer {
            let cols = UInt(cellsPerBlock.x * orientations)
            let rows = UInt(cellsPerBlock.y)
            
            let ta = UInt(numberOfCellX*orientations)
            let tc = UInt(cellsPerBlock.x*orientations)
            
            let blockSize = cellsPerBlock.y * cellsPerBlock.x * orientations
            var sum: Double = 0
            
            var blockHead = $0.baseAddress!
            for by in 0..<numberOfBlocksY {
                for bx in 0..<numberOfBlocksX {
                    let cellHeadIndex = (by * numberOfCellX + bx) * orientations
                    
                    // copy block
                    vDSP_mmovD(&histograms + cellHeadIndex,
                               blockHead,
                               cols, rows,
                               ta, tc)
                    
                    // normalize block
                    switch normalization {
                    case .l1:
                        vDSP_sveD(blockHead, 1, &sum, UInt(blockSize))
                        sum += eps
                        vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                    case .l1sqrt:
                        vDSP_sveD(blockHead, 1, &sum, UInt(blockSize))
                        sum += eps
                        vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                        var _cnt = Int32(blockSize)
                        vvsqrt(blockHead, blockHead, &_cnt)
                    case .l2:
                        vDSP_svesqD(blockHead, 1, &sum, UInt(blockSize))
                        sum = sqrt(sum) + eps
                        vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                    case .l2Hys:
                        vDSP_svesqD(blockHead, 1, &sum, UInt(blockSize))
                        sum = sqrt(sum) + eps
                        vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                        
                        var lower = 0.0
                        var upper = 0.2
                        vDSP_vclipD(blockHead, 1, &lower, &upper, blockHead, 1, UInt(blockSize))
                        
                        vDSP_svesqD(blockHead, 1, &sum, UInt(blockSize))
                        sum = sqrt(sum) + eps
                        vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                    }
                    
                    blockHead += blockSize
                }
            }
        }
        
        return normalizedHistogram
    }
}

