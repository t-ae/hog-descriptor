import XCTest
import HOGDescriptor

final class HOGFeatureExtractorTests: XCTestCase {
    func testScaleInvariance() {
        do {
            let extractor = HOGDescriptor(orientations: 9, cellSpan: 3, blockSpan: 4, normalization: .l1)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = extractor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let extractor = HOGDescriptor(orientations: 9,
                                          pixelsPerCell: (2, 4),
                                          cellsPerBlock: (3, 4),
                                          normalization: .l1,
                                          transformSqrt: true)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = extractor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let extractor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l1sqrt)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = extractor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let extractor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = extractor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let extractor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2Hys)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = extractor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
    }
    
    func testPerformanceL1() {
        let extractor = HOGDescriptor(orientations: 9, cellSpan: 8, blockSpan: 3, normalization: .l1)
        
        let iterations = 100
        let width = 256
        let height = 256
        let image = (0..<width*height).map { _ in Double.random(in: 0..<255) }
        
        measure {
            for _ in 0..<iterations {
                _ = extractor.getDescriptor(data: image, width: width, height: height)
            }
        }
    }
    
    func testPerformanceL1sqrt() {
        let extractor = HOGDescriptor(orientations: 9, cellSpan: 8, blockSpan: 3, normalization: .l1sqrt)
        
        let iterations = 100
        let width = 256
        let height = 256
        let image = (0..<width*height).map { _ in Double.random(in: 0..<255) }
        
        measure {
            for _ in 0..<iterations {
                _ = extractor.getDescriptor(data: image, width: width, height: height)
            }
        }
    }
    
    func testPerformanceL2() {
        let extractor = HOGDescriptor(orientations: 9, cellSpan: 8, blockSpan: 3, normalization: .l2)
        
        let iterations = 100
        let width = 256
        let height = 256
        let image = (0..<width*height).map { _ in Double.random(in: 0..<255) }
        
        measure {
            for _ in 0..<iterations {
                _ = extractor.getDescriptor(data: image, width: width, height: height)
            }
        }
    }
    
    func testPerformanceL2Hys() {
        let extractor = HOGDescriptor(orientations: 9, cellSpan: 8, blockSpan: 3, normalization: .l2Hys)
        
        let iterations = 100
        let width = 256
        let height = 256
        let image = (0..<width*height).map { _ in Double.random(in: 0..<255) }
        
        measure {
            for _ in 0..<iterations {
                _ = extractor.getDescriptor(data: image, width: width, height: height)
            }
        }
    }
    
    static var allTests = [
        ("testScaleInvariance", testScaleInvariance),
        ]
}
