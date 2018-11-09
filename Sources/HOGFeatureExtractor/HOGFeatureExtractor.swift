import Foundation
import Accelerate

public class HOGFeatureExtractor {
    
    public enum NormalizationMethod {
        case l1, l2
    }
    
    public let pixelsPerCell: (x: Int, y: Int)
    public let cellsPerBlock: (x: Int, y: Int)
    public let orientation: Int
    
    public let normalization: NormalizationMethod
    
    public let eps = 1e-5
    
    /// Create HOGFeatureExtractor.
    /// - Parameters:
    ///   - pixelsPerCell: Size (in pixels) of a cell.
    ///   - cellsPerBlock: Number of cells in each block.
    ///   - orientation: Number of orientation bins. default: 9
    ///   - normalization: Block normalization method. default: .l1
    public init(pixelsPerCell: (x: Int, y: Int),
                cellsPerBlock: (x: Int, y: Int),
                orientation: Int = 9,
                normalization: NormalizationMethod = .l1) {
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.orientation = orientation
        self.normalization = normalization
    }
    
    /// Create HOGFeatureExtractor.
    /// - Parameters:
    ///   - cellSpan: Size (in pixels) of a cell.
    ///   - blockSpan: Number of cells in each block.
    ///   - orientation: Number of orientation bins. default: 9
    ///   - normalization: Block normalization method. default: .l1
    public convenience init(cellSpan: Int,
                            blockSpan: Int,
                            orientation: Int = 9,
                            normalization: NormalizationMethod = .l1) {
        self.init(pixelsPerCell: (cellSpan, cellSpan),
                  cellsPerBlock: (blockSpan, blockSpan),
                  orientation: orientation,
                  normalization: normalization)
    }
    
    private func derivate(data: UnsafePointer<Double>, width: Int, height: Int) -> (x: [Double], y: [Double]) {
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hog.py#L23-L44
        
        var xImg = [Double](repeating: 0, count: width*height)
        
        xImg.withUnsafeMutableBufferPointer {
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
        
        var yImg = [Double](repeating: 0, count: width*height)
        yImg.withUnsafeMutableBufferPointer {
            let yImgPtr = $0.baseAddress!
            
            let dpUp = data
            let dpDown = data.advanced(by: 2*width)
            let dst = yImgPtr.advanced(by: width)
            
            vDSP_vsubD(dpUp, 1, dpDown, 1, dst, 1, UInt(width*(height-2)))
        }
        
        return (xImg, yImg)
    }
    
    /// Extract HOG Feature from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    public func extract(data: UnsafePointer<Double>,
                        width: Int,
                        height: Int) -> [Double] {
        
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        // derivatives
        let (gradX, gradY) = derivate(data: data, width: width, height: height)
        
        // calculate gradient directions and intensities
        var grad = [Double](repeating: 0, count: gradX.count)
        var _cnt = Int32(grad.count)
        vvatan2(&grad, gradY, gradX, &_cnt) // [-pi, pi]
        var multiplier = Double(orientation) / .pi
        var adder = Double(orientation)
        vDSP_vsmsaD(grad, 1, &multiplier, &adder, &grad, 1, UInt(grad.count)) // [0, 2*orientation]
        
        var magnitude = [Double](repeating: 0, count: gradX.count)
        vDSP_vdistD(gradX, 1, gradY, 1, &magnitude, 1, UInt(magnitude.count))
        
        // accumulate to histogram
        var histograms = [Double](repeating: 0, count: numberOfCellY*numberOfCellX*orientation)
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
                while directionIndex >= orientation {
                    directionIndex -= orientation
                }
                
                let histogramIndex = (cellY * numberOfCellX + cellX) * orientation
                histograms[histogramIndex + directionIndex] += magnitude[y*width+x]
            }
        }
        
        // normalize
        let numberOfBlocksX = numberOfCellX - cellsPerBlock.x + 1
        let numberOfBlocksY = numberOfCellY - cellsPerBlock.y + 1
        
        let featureCount = numberOfBlocksY*numberOfBlocksX*cellsPerBlock.y*cellsPerBlock.x*orientation
        var normalizedHistogram = [Double](repeating: 0, count: featureCount)
        
        for by in 0..<numberOfBlocksY {
            for bx in 0..<numberOfBlocksX {
                let blockHead = ((by*numberOfBlocksX) + bx) * cellsPerBlock.y * cellsPerBlock.x * orientation
                
                for cy in 0..<cellsPerBlock.y {
                    let size = cellsPerBlock.x * orientation
                    let blockRowHead = blockHead + cy * cellsPerBlock.x * orientation
                    let cellHead = (by + cy) * numberOfCellX * orientation
                    
                    normalizedHistogram.withUnsafeMutableBufferPointer {
                        let head = $0.baseAddress! + blockRowHead
                        memcpy(head,
                               &histograms + (cellHead * MemoryLayout<Double>.size),
                               size * MemoryLayout<Double>.size)
                    }
                }
                
                normalizedHistogram.withUnsafeMutableBufferPointer {
                    let head = $0.baseAddress! + blockHead
                    let size = cellsPerBlock.y * cellsPerBlock.x * orientation
                    switch normalization {
                    case .l1:
                        var sum: Double = 0
                        vDSP_sveD(head, 1, &sum, UInt(size))
                        sum += eps
                        vDSP_vsdivD(head, 1, &sum, head, 1, UInt(size))
                    case .l2:
                        var sum2: Double = 0
                        vDSP_svesqD(head, 1, &sum2, UInt(size))
                        sum2 = sqrt(sum2) + eps
                        vDSP_vsdivD(head, 1, &sum2, head, 1, UInt(size))
                    }
                }
            }
        }
        
        return normalizedHistogram
    }
    
    /// Extract HOG Feature from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    public func extract(data: UnsafePointer<UInt8>, width: Int, height: Int) -> [Double] {
        var doubleImage = [Double](repeating: 0, count: width*height)
        vDSP_vfltu8D(data, 1, &doubleImage, 1, UInt(width*height))
        
        return extract(data: doubleImage, width: width, height: height)
    }
}

