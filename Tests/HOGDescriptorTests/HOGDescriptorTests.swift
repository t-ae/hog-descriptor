import XCTest
import HOGDescriptor

final class HOGDescriptorTests: XCTestCase {
    func testGetSize() {
        do {
            let width = 16
            let height = 16
            let hogDescriptor = HOGDescriptor(orientations: 9,
                                              pixelsPerCell: (4, 4),
                                              cellsPerBlock: (3, 3))
            let image = [Double](repeating: 0, count: width * height)
            XCTAssertEqual(hogDescriptor.getDescriptorSize(width: width, height: height),
                           hogDescriptor.getDescriptor(data: image, width: width, height: height).count)
        }
        do {
            let width = 8
            let height = 16
            let hogDescriptor = HOGDescriptor(orientations: 9,
                                              pixelsPerCell: (3, 2),
                                              cellsPerBlock: (3, 2))
            let image = [Double](repeating: 0, count: width * height)
            XCTAssertEqual(hogDescriptor.getDescriptorSize(width: width, height: height),
                           hogDescriptor.getDescriptor(data: image, width: width, height: height).count)
        }
    }
    func testScaleInvariance() {
        do {
            let hogDescriptor = HOGDescriptor(orientations: 9, cellSpan: 3, blockSpan: 4, normalization: .l1)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 9,
                                              pixelsPerCell: (2, 4),
                                              cellsPerBlock: (3, 4),
                                              normalization: .l1,
                                              transformSqrt: true)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l1sqrt)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2Hys)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
    }
    
    
    static var allTests = [
        ("testScaleInvariance", testScaleInvariance),
    ]
}
