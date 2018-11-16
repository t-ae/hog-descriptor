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
    
    /// Get necessary workspace size for specified width/size image.
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
    ///   - data: Pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Precondition:
    ///   - data.count == width*height
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: [Double],
                              width: Int,
                              height: Int) -> [Double] {
        return data.withUnsafeBufferPointer {
            getDescriptor(data: $0, width: width, height: height)
        }
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Precondition:
    ///   - data.count == width*height
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: UnsafeBufferPointer<Double>,
                              width: Int,
                              height: Int) -> [Double] {
        var descriptor = [Double](repeating: 0, count: getDescriptorSize(width: width, height: height))
        var workspace = [Double](repeating: 0, count: getWorkspaceSize(width: width, height: height))
        
        if transformSqrt {
            var transformed = [Double](repeating: 0, count: data.count)
            var count = Int32(transformed.count)
            vvsqrt(&transformed, data.baseAddress!, &count)
            getDescriptor(data: .init(start: transformed, count: transformed.count),
                          width: width, height: height,
                          output: .init(start: &descriptor, count: descriptor.count),
                          workspace: .init(start: &workspace, count: workspace.count))
        } else {
            getDescriptor(data: data, width: width, height: height,
                          output: .init(start: &descriptor, count: descriptor.count),
                          workspace: .init(start: &workspace, count: workspace.count))
        }
        
        return descriptor
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Precondition:
    ///   - data.count == width*height
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: [UInt8],
                              width: Int,
                              height: Int) -> [Double] {
        return data.withUnsafeBufferPointer {
            getDescriptor(data: $0, width: width, height: height)
        }
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    /// - Precondition:
    ///   - data.count == width*height
    /// - Returns: HOG feature vector (raveled N-D array of [NumBlocksY, NumBlocksX, CellsPerBlockY, CellsPerBlockX, Orientations]).
    public func getDescriptor(data: UnsafeBufferPointer<UInt8>, width: Int, height: Int) -> [Double] {
        var doubleImage = [Double](repeating: 0, count: width*height)
        vDSP_vfltu8D(data.baseAddress!, 1, &doubleImage, 1, UInt(doubleImage.count))
        
        if transformSqrt {
            var count = Int32(doubleImage.count)
            vvsqrt(&doubleImage, doubleImage, &count)
        }
        
        var descriptor = [Double](repeating: 0, count: getDescriptorSize(width: width, height: height))
        var workspace = [Double](repeating: 0, count: getWorkspaceSize(width: width, height: height))
        getDescriptor(data: .init(start: doubleImage, count: doubleImage.count),
                      width: width, height: height,
                      output: .init(start: &descriptor, count: descriptor.count),
                      workspace: .init(start: &workspace, count: workspace.count))
        
        return descriptor
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Note: This method itself doesn't allocate memories.
    /// So if you calculate HOG repeatedly, you can gain slight performance improvement with this.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    ///   - output: output of HOG descriptor.
    ///   - workspace: workspace.
    /// - Precondition:
    ///   - data.count == width*height
    ///   - output.count >= getDescriptorSize(width: width, height: height)
    ///   - workspace.count >= getWorkspaceSize(width: width, height: height)
    public func getDescriptor(data: UnsafeBufferPointer<Double>,
                              width: Int,
                              height: Int,
                              output: UnsafeMutableBufferPointer<Double>,
                              workspace: UnsafeMutableBufferPointer<Double>) {
        let workspaceSize = getWorkspaceSize(width: width, height: height)
        let descriptorSize = getDescriptorSize(width: width, height: height)
        
        precondition(data.count == width*height)
        precondition(output.count >= descriptorSize)
        precondition(workspace.count >= workspaceSize)
        
        // 0 clear
        memset(workspace.baseAddress!, 0, workspaceSize*MemoryLayout<Double>.size)
        
        // derivatives
        let gradSize = width*height
        let gradY = UnsafeMutableBufferPointer(rebasing: workspace[start: gradSize, count: gradSize])
        let gradX = UnsafeMutableBufferPointer(rebasing: workspace[start: gradSize*2, count: gradSize])
        differentiate(data: data, width: width, height: height, gradX: gradX, gradY: gradY)
        
        // calculate gradient directions and magnitudes
        let grad = UnsafeMutableBufferPointer(rebasing: workspace[start: 0, count: gradSize])
        do {
            var _cnt = Int32(gradSize)
            vvatan2(grad.baseAddress!, gradY.baseAddress!, gradX.baseAddress!, &_cnt) // [-pi, pi]
            var multiplier = Double(orientations) / .pi
            var adder = Double(orientations)
            vDSP_vsmsaD(grad.baseAddress!, 1,
                        &multiplier, &adder,
                        grad.baseAddress!, 1,
                        UInt(gradSize)) // [0, 2*orientation]
        }
        
        let magnitude = UnsafeMutableBufferPointer(rebasing: workspace[start: gradSize, count: gradSize])
        vDSP_vdistD(gradY.baseAddress!, 1,
                    gradX.baseAddress!, 1,
                    magnitude.baseAddress!, 1,
                    UInt(gradSize))
        
        
        // accumulate to histograms
        
        // N-D array of [numberOfCells.y, numberOfCells.x, orientations]
        let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
        let histogramsSize = numberOfCells.y * numberOfCells.x * orientations
        let histograms = UnsafeMutableBufferPointer(rebasing: workspace[start: (gradSize*2),
                                                                        count: histogramsSize])
        
        // 0 clear
        memset(histograms.baseAddress!, 0, histogramsSize*MemoryLayout<Double>.size)
        
        // weighted vote
        for cellY in 0..<numberOfCells.y {
            for cellX in 0..<numberOfCells.x {
                let headIndex = (cellY * numberOfCells.x + cellX) * orientations
                let histogramHead = UnsafeMutableBufferPointer(rebasing: histograms[headIndex...])
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
        vDSP_vsdivD(histograms.baseAddress!, 1,
                    &divisor,
                    histograms.baseAddress!, 1,
                    UInt(histogramsSize))
        
        // normalize
        let numberOfBlocks = (x: numberOfCells.x - cellsPerBlock.x + 1,
                              y: numberOfCells.y - cellsPerBlock.y + 1)
        
        // N-D array of [numberOfBlocks.y, numberOfBlocks.x, cellsPerBlock.y, cellsPerBlock.x, orientations]
        
        normalize(histograms: UnsafeBufferPointer(histograms),
                  numberOfCells: numberOfCells,
                  blocks: .init(rebasing: output[..<descriptorSize]),
                  numberOfBlocks: numberOfBlocks)
    }
    
    /// Calculate vertical/horizontal differentials.
    func differentiate(data: UnsafeBufferPointer<Double>,
                       width: Int,
                       height: Int,
                       gradX: UnsafeMutableBufferPointer<Double>,
                       gradY: UnsafeMutableBufferPointer<Double>) {
        // https://github.com/scikit-image/scikit-image/blob/9c4632f43eb6f6e85bf33f9adf8627d01b024496/skimage/feature/_hog.py#L23-L44
        
        assert(gradX.count == width*height)
        assert(gradY.count == width*height)
        
        do {
            var dpLeft = data.baseAddress!
            var dpRight = data.baseAddress!.advanced(by: 2)
            var dst = gradX.baseAddress!.advanced(by: 1)
            for _ in 0..<height {
                dst.advanced(by: -1).pointee = 0
                vDSP_vsubD(dpLeft, 1, dpRight, 1, dst, 1, UInt(width-2))
                
                dpLeft += width
                dpRight += width
                dst += width
            }
        }
        
        do {
            let dpUp = data.baseAddress!
            let dpDown = data.baseAddress!.advanced(by: 2*width)
            let dst = gradY.baseAddress!.advanced(by: width)
            
            vDSP_vsubD(dpUp, 1, dpDown, 1, dst, 1, UInt(width*(height-2)))
        }
    }
    
    /// Copy and Normalize blocks.
    func normalize(histograms: UnsafeBufferPointer<Double>,
                   numberOfCells: (x: Int, y: Int),
                   blocks: UnsafeMutableBufferPointer<Double>,
                   numberOfBlocks: (x: Int ,y: Int)) {
        
        assert(histograms.count == numberOfCells.y*numberOfCells.x*orientations)
        assert(blocks.count == numberOfBlocks.y*numberOfBlocks.x*cellsPerBlock.y*cellsPerBlock.x*orientations)
        
        let cols = UInt(cellsPerBlock.x * orientations)
        let rows = UInt(cellsPerBlock.y)
        
        let ta = UInt(numberOfCells.x*orientations)
        let tc = UInt(cellsPerBlock.x*orientations)
        
        let blockSize = cellsPerBlock.y * cellsPerBlock.x * orientations
        var sum: Double = 0
        
        var blockHead = blocks
        for by in 0..<numberOfBlocks.y {
            for bx in 0..<numberOfBlocks.x {
                let histHeadIndex = (by * numberOfCells.x + bx) * orientations
                
                // copy block
                vDSP_mmovD(histograms.baseAddress! + histHeadIndex,
                           blockHead.baseAddress!,
                           cols, rows,
                           ta, tc)
                
                // normalize block
                switch normalization {
                case .l1:
                    vDSP_sveD(blockHead.baseAddress!, 1, &sum, UInt(blockSize))
                    sum += eps
                    vDSP_vsdivD(blockHead.baseAddress!, 1, &sum,
                                blockHead.baseAddress!, 1, UInt(blockSize))
                case .l1sqrt:
                    vDSP_sveD(blockHead.baseAddress!, 1, &sum, UInt(blockSize))
                    sum += eps
                    vDSP_vsdivD(blockHead.baseAddress!, 1, &sum,
                                blockHead.baseAddress!, 1, UInt(blockSize))
                    var _cnt = Int32(blockSize)
                    vvsqrt(blockHead.baseAddress!, blockHead.baseAddress!, &_cnt)
                case .l2:
                    vDSP_svesqD(blockHead.baseAddress!, 1, &sum, UInt(blockSize))
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead.baseAddress!, 1, &sum,
                                blockHead.baseAddress!, 1, UInt(blockSize))
                case .l2Hys:
                    vDSP_svesqD(blockHead.baseAddress!, 1, &sum, UInt(blockSize))
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead.baseAddress!, 1, &sum,
                                blockHead.baseAddress!, 1,
                                UInt(blockSize))
                    
                    var lower = 0.0
                    var upper = 0.2
                    vDSP_vclipD(blockHead.baseAddress!, 1,
                                &lower, &upper,
                                blockHead.baseAddress!, 1,
                                UInt(blockSize))
                    
                    vDSP_svesqD(blockHead.baseAddress!, 1, &sum, UInt(blockSize))
                    sum = sqrt(sum + eps*eps)
                    vDSP_vsdivD(blockHead.baseAddress!, 1, &sum,
                                blockHead.baseAddress!, 1,
                                UInt(blockSize))
                }
                
                blockHead = UnsafeMutableBufferPointer(rebasing: blockHead[blockSize...])
            }
        }
    }
    
    /// Dump workspace blocks.
    private func dumpWorkspace(_ workspace: UnsafePointer<Double>, width: Int, height: Int) {
        let block1 = [Double](UnsafeBufferPointer(start: workspace, count: width*height))
        let block2 = [Double](UnsafeBufferPointer(start: workspace + width*height, count: width*height))
        let block3 = [Double](UnsafeBufferPointer(start: workspace + 2*width*height, count: getWorkspaceSize(width: width, height: height) - 2*width*height))
        
        print(block1)
        print(block2)
        print(block3)
    }
}

extension UnsafeBufferPointer {
    subscript(start start: Int, count count: Int) -> Slice<UnsafeBufferPointer<Element>> {
        return self[start..<start+count]
    }
}

extension UnsafeMutableBufferPointer {
    subscript(start start: Int, count count: Int) -> Slice<UnsafeMutableBufferPointer<Element>> {
        return self[start..<start+count]
    }
}
