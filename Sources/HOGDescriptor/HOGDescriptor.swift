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
    /// - Precondition: orientations*2 <= UInt8.max (Internally using UInt8 for voting)
    public init(orientations: Int = 9,
                pixelsPerCell: (x: Int, y: Int) = (8, 8),
                cellsPerBlock: (x: Int, y: Int) = (3, 3),
                normalization: NormalizationMethod = .l1,
                transformSqrt: Bool = false) {
        precondition(orientations*2 <= UInt8.max)
        
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
    /// - Precondition: orientation*2 <= UInt8.max (Internally using UInt8 for voting)
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
    
    /// Get size of HOG descriptor for specified width/height image.
    public func getDescriptorSize(width: Int, height: Int) -> Int {
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        let numberOfBlocksX = numberOfCellX - cellsPerBlock.x + 1
        let numberOfBlocksY = numberOfCellY - cellsPerBlock.y + 1
        
        return numberOfBlocksY*numberOfBlocksX*cellsPerBlock.y*cellsPerBlock.x*orientations
    }
    
    /// Get necessary workspace sizes for specified width/height image.
    public func getWorkspaceSizes(width: Int, height: Int) -> (double: Int, uint8: Int) {
        let numberOfCellX = width / pixelsPerCell.x
        let numberOfCellY = height / pixelsPerCell.y
        
        let gradSize = width*height
        // gradSize == gradXSize == gradYSize == magnitudeSize
        
        let histogramsSize = numberOfCellY*numberOfCellX*orientations
        
        // 1. [empty, gradY, gradX]
        // 2. [grad, gradY, gradX]
        // 3. [magnitude, gradY, gradX]
        // 4. [magnitude, histograms]
        return (max(3*gradSize, gradSize + histogramsSize), gradSize)
    }
    
    /// Create workspaces.
    public func createWorkspaces(width: Int, height: Int) -> (double: [Double], uint8: [UInt8]) {
        let sizes = getWorkspaceSizes(width: width, height: height)
        return ([Double](repeating: 0, count: sizes.double),
                [UInt8](repeating: 0, count: sizes.uint8))
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
        var workspaces = createWorkspaces(width: width, height: height)
        
        if transformSqrt {
            var transformed = [Double](repeating: 0, count: data.count)
            var count = Int32(transformed.count)
            vvsqrt(&transformed, data.baseAddress!, &count)
            getDescriptor(data: .init(start: transformed, count: transformed.count),
                          width: width, height: height,
                          descriptor: .init(start: &descriptor, count: descriptor.count),
                          workspaces: &workspaces)
        } else {
            getDescriptor(data: data, width: width, height: height,
                          descriptor: .init(start: &descriptor, count: descriptor.count),
                          workspaces: &workspaces)
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
        var workspaces = createWorkspaces(width: width, height: height)
        getDescriptor(data: .init(start: doubleImage, count: doubleImage.count),
                      width: width, height: height,
                      descriptor: .init(start: &descriptor, count: descriptor.count),
                      workspaces: &workspaces)
        
        return descriptor
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Note: This method itself doesn't allocate memories.
    /// So if you calculate HOG repeatedly, you can gain slight performance improvement with this.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    ///   - descriptor: output of HOG descriptor.
    ///   - workspaces: Tuple of workspaces.
    /// - Precondition:
    ///   - data.count == width*height
    ///   - output.count >= getDescriptorSize(width: width, height: height)
    ///   - workspaces.double.count >= getWorkspaceSize(width: width, height: height).double
    ///   - workspaces.uint8.count >= getWorkspaceSize(width: width, height: height).uint8
    public func getDescriptor(data: UnsafeBufferPointer<Double>,
                              width: Int,
                              height: Int,
                              descriptor: UnsafeMutableBufferPointer<Double>,
                              workspaces: inout (double: [Double], uint8: [UInt8])) {
        getDescriptor(data: data,
                      width: width,
                      height: height,
                      descriptor: descriptor,
                      workspace1: .init(start: &workspaces.double, count: workspaces.double.count),
                      workspace2: .init(start: &workspaces.uint8, count: workspaces.uint8.count))
    }
    
    /// Get HOG descriptor from gray scale image.
    /// - Note: This method itself doesn't allocate memories.
    /// So if you calculate HOG repeatedly, you can gain slight performance improvement with this.
    /// - Parameters:
    ///   - data: Head of pixel values of gray scale image, row major order.
    ///   - width: Width of image.
    ///   - height: Height of image.
    ///   - descriptor: output of HOG descriptor.
    ///   - workspace1: Double workspace.
    ///   - workspace2: UInt8 workspace.
    /// - Precondition:
    ///   - data.count == width*height
    ///   - output.count >= getDescriptorSize(width: width, height: height)
    ///   - workspace1.count >= getWorkspaceSize(width: width, height: height).double
    ///   - workspace2.count >= getWorkspaceSize(width: width, height: height).uint8
    public func getDescriptor(data: UnsafeBufferPointer<Double>,
                              width: Int,
                              height: Int,
                              descriptor: UnsafeMutableBufferPointer<Double>,
                              workspace1: UnsafeMutableBufferPointer<Double>,
                              workspace2: UnsafeMutableBufferPointer<UInt8>) {
        let workspaceSize = getWorkspaceSizes(width: width, height: height)
        let descriptorSize = getDescriptorSize(width: width, height: height)
        
        precondition(data.count == width*height)
        precondition(descriptor.count >= descriptorSize)
        precondition(workspace1.count >= workspaceSize.double)
        precondition(workspace2.count >= workspaceSize.uint8)
        
        // 0 clear
        memset(workspace1.baseAddress!, 0, workspaceSize.double*MemoryLayout<Double>.size)
        
        // differentiate
        let gradSize = width*height
        let gradY = UnsafeMutableBufferPointer(rebasing: workspace1[start: gradSize, count: gradSize])
        let gradX = UnsafeMutableBufferPointer(rebasing: workspace1[start: gradSize*2, count: gradSize])
        differentiate(data: data, width: width, height: height, gradX: gradX, gradY: gradY)
        
        // Calculate gradient directions
        // These will be casted to Int while histogram step.
        // But precasting with vDSP is a bit faster.
        let grad = workspace2
        do {
            let gradD = UnsafeMutableBufferPointer(rebasing: workspace1[start: 0, count: gradSize])
            var _cnt = Int32(gradSize)
            vvatan2(gradD.baseAddress!, gradY.baseAddress!, gradX.baseAddress!, &_cnt) // [-pi, pi]
            var multiplier = Double(orientations) / .pi
            var adder = Double(orientations)
            vDSP_vsmsaD(gradD.baseAddress!, 1,
                        &multiplier, &adder,
                        gradD.baseAddress!, 1,
                        UInt(gradSize)) // [0, 2*orientation]
            vDSP_vfixu8D(gradD.baseAddress!, 1, grad.baseAddress!, 1, UInt(gradSize))
        }
        
        // Calculate magnitudes
        let magnitude = UnsafeMutableBufferPointer(rebasing: workspace1[start: 0, count: gradSize])
        vDSP_vdistD(gradY.baseAddress!, 1,
                    gradX.baseAddress!, 1,
                    magnitude.baseAddress!, 1,
                    UInt(gradSize))
        
        
        // accumulate to histograms
        
        // N-D array of [numberOfCells.y, numberOfCells.x, orientations]
        let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
        let histogramsSize = numberOfCells.y * numberOfCells.x * orientations
        let histograms = UnsafeMutableBufferPointer(rebasing: workspace1[start: gradSize,
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
                        let index = y*width + x
                        var directionIndex = Int(grad[index])
                        while directionIndex >= orientations {
                            directionIndex -= orientations
                        }
                        histogramHead[directionIndex] += magnitude[index]
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
                  blocks: .init(rebasing: descriptor[..<descriptorSize]),
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
        assert(gradX.allSatisfy { $0 == 0 })
        assert(gradY.allSatisfy { $0 == 0 })
        
        do {
            var dpLeft = data.baseAddress!
            var dpRight = data.baseAddress!.advanced(by: 2)
            var dst = gradX.baseAddress!.advanced(by: 1)
            
            for _ in 0..<height {
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
