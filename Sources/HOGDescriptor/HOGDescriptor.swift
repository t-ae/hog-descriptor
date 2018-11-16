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
    
    /// Create HOGDescriptor.
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
    
    /// Create HOGDescriptor with square cells/blocks.
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
    
    public func getWorkspaceSize(width: Int, height: Int) -> Int {
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        let gradSize = width*height
        // gradSize == gradXSize == gradYSize == magnitudeSize
        
        let histogramsSize = numberOfCellY*numberOfCellX*orientations
        
        // 1. [empty, gradY, gradX]
        // 2. [grad, gradY, gradX]
        // 3. [grad, magnitude, gradX]
        // 4. [grad, magnitude, histograms]
        
        return max(3*gradSize, 2*gradSize + histogramsSize)
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: UnsafePointer<Double>,
                              width: Int,
                              height: Int) -> [Double] {
        var descriptor = [Double](repeating: 0, count: getDescriptorSize(width: width, height: height))
        var workspace = [Double](repeating: 0, count: getWorkspaceSize(width: width, height: height))
        if transformSqrt {
            var transformed = [Double](repeating: 0, count: width*height)
            var count = Int32(transformed.count)
            vvsqrt(&transformed, data, &count)
            _getDescriptor(data: transformed, width: width, height: height,
                           output: &descriptor, workspace: &workspace)
        } else {
            _getDescriptor(data: data, width: width, height: height,
                           output: &descriptor, workspace: &workspace)
        }
        return descriptor
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: UnsafePointer<UInt8>, width: Int, height: Int) -> [Double] {
        var doubleImage = [Double](repeating: 0, count: width*height)
        vDSP_vfltu8D(data, 1, &doubleImage, 1, UInt(doubleImage.count))
        if transformSqrt {
            var count = Int32(doubleImage.count)
            vvsqrt(&doubleImage, doubleImage, &count)
        }
        var descriptor = [Double](repeating: 0, count: getDescriptorSize(width: width, height: height))
        var workspace = [Double](repeating: 0, count: getWorkspaceSize(width: width, height: height))
        _getDescriptor(data: doubleImage, width: width, height: height,
                       output: &descriptor, workspace: &workspace)
        
        return descriptor
    }
    
    func derivate(data: UnsafePointer<Double>,
                  width: Int,
                  height: Int,
                  gradX: UnsafeMutablePointer<Double>,
                  gradY: UnsafeMutablePointer<Double>) {
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hog.py#L23-L44
        
        // [empty, gradY, gradX]
        do {
            var dpLeft = data
            var dpRight = data.advanced(by: 2)
            var dst = gradX.advanced(by: 1)
            for _ in 0..<height {
                dst.advanced(by: -1).pointee = 0
                vDSP_vsubD(dpLeft, 1, dpRight, 1, dst, 1, UInt(width-2))
                
                dpLeft += width
                dpRight += width
                dst += width
            }
        }
        
        do {
            let dpUp = data
            let dpDown = data.advanced(by: 2*width)
            let dst = gradY.advanced(by: width)
            
            vDSP_vsubD(dpUp, 1, dpDown, 1, dst, 1, UInt(width*(height-2)))
        }
    }
    
    func normalize(histograms: UnsafePointer<Double>,
                   numberOfCells: (x: Int, y: Int),
                   blocks: UnsafeMutablePointer<Double>,
                   numberOfBlocks: (x: Int ,y: Int)) {
        let cols = UInt(cellsPerBlock.x * orientations)
        let rows = UInt(cellsPerBlock.y)
        
        let ta = UInt(numberOfCells.x*orientations)
        let tc = UInt(cellsPerBlock.x*orientations)
        
        let blockSize = cellsPerBlock.y * cellsPerBlock.x * orientations
        var sum: Double = 0
        
        var blockHead = blocks
        for by in 0..<numberOfBlocks.y {
            for bx in 0..<numberOfBlocks.x {
                let cellHeadIndex = (by * numberOfCells.x + bx) * orientations
                
                // copy block
                vDSP_mmovD(histograms + cellHeadIndex,
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
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                case .l2Hys:
                    vDSP_svesqD(blockHead, 1, &sum, UInt(blockSize))
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                    
                    var lower = 0.0
                    var upper = 0.2
                    vDSP_vclipD(blockHead, 1, &lower, &upper, blockHead, 1, UInt(blockSize))
                    
                    vDSP_svesqD(blockHead, 1, &sum, UInt(blockSize))
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead, 1, &sum, blockHead, 1, UInt(blockSize))
                }
                
                blockHead += blockSize
            }
        }
    }
    
    public func _getDescriptor(data: UnsafePointer<Double>,
                               width: Int,
                               height: Int,
                               output: UnsafeMutablePointer<Double>,
                               workspace: UnsafeMutablePointer<Double>) {
        // 0 clear
        let workspaceSize = getWorkspaceSize(width: width, height: height)
        memset(workspace, 0, workspaceSize*MemoryLayout<Double>.size)
        
        let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
        
        let gradSize = width*height
        
        // derivatives
        let gradY = workspace + gradSize
        let gradX = gradY + gradSize
        derivate(data: data, width: width, height: height, gradX: gradX, gradY: gradY)
        
        // calculate gradient directions and magnitudes
        let grad = workspace
        do {
            var _cnt = Int32(gradSize)
            vvatan2(grad, gradY, gradX, &_cnt) // [-pi, pi]
            var multiplier = Double(orientations) / .pi
            var adder = Double(orientations)
            vDSP_vsmsaD(grad, 1, &multiplier, &adder, grad, 1, UInt(gradSize)) // [0, 2*orientation]
        }
        
        let magnitude = workspace + gradSize
        vDSP_vdistD(gradY, 1, gradX, 1, magnitude, 1, UInt(gradSize))
        
        
        // accumulate to histograms
        
        // N-D array of [numberOfCells.y, numberOfCells.x, orientations]
        let histograms = workspace + 2*gradSize
        let histogramsSize = numberOfCells.y * numberOfCells.x * orientations
        memset(histograms, 0, histogramsSize*MemoryLayout<Double>.size)
        
        for cellY in 0..<numberOfCells.y {
            for cellX in 0..<numberOfCells.x {
                let histogramHead = histograms + (cellY * numberOfCells.x + cellX) * orientations
                for y in cellY*pixelsPerCell.y..<(cellY+1)*pixelsPerCell.y {
                    for x in cellX*pixelsPerCell.x..<(cellX+1)*pixelsPerCell.x {
                        var directionIndex = Int(grad[y*width+x])
                        while directionIndex >= orientations {
                            directionIndex -= orientations
                        }
                        histogramHead[directionIndex] += magnitude[y*width+x]
                    }
                }
            }
        }
        
        // Scale histograms
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hoghistogram.pyx#L74
        // Basically it's helpful only for visualization.
        // But, since we add `eps` while normalization, the result will have slight differences from skimage's without this.
        var divisor = Double(pixelsPerCell.y * pixelsPerCell.x)
        vDSP_vsdivD(histograms, 1, &divisor, histograms, 1, UInt(histogramsSize))
        
        // normalize
        let numberOfBlocks = (x: numberOfCells.x - cellsPerBlock.x + 1,
                              y: numberOfCells.y - cellsPerBlock.y + 1)
        
        // N-D array of [numberOfBlocks.y, numberOfBlocks.x, cellsPerBlock.y, cellsPerBlock.x, orientations]
        normalize(histograms: histograms, numberOfCells: numberOfCells,
                  blocks: output, numberOfBlocks: numberOfBlocks)
    }
    
    private func dumpWorkspace(_ workspace: UnsafePointer<Double>, width: Int, height: Int) {
        let block1 = [Double](UnsafeBufferPointer(start: workspace, count: width*height))
        let block2 = [Double](UnsafeBufferPointer(start: workspace + width*height, count: width*height))
        let block3 = [Double](UnsafeBufferPointer(start: workspace + 2*width*height, count: getWorkspaceSize(width: width, height: height) - 2*width*height))
        
        print(block1)
        print(block2)
        print(block3)
    }
}

