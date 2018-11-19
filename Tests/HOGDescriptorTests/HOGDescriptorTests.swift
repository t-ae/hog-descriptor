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
        let image1 = (0..<64*64).map { _ in Double(UInt8.random(in: 0...255)) }
        let image2 = image1.map { Double($0)/255 }
        
        do {
            let hogDescriptor = HOGDescriptor(orientations: 9, cellSpan: 3, blockSpan: 4, normalization: .l1)
            
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
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l1sqrt)
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2)
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
        do {
            let hogDescriptor = HOGDescriptor(orientations: 5, cellSpan: 8, blockSpan: 4, normalization: .l2Hys)
            
            let f1 = hogDescriptor.getDescriptor(data: image1, width: 64, height: 64)
            let f2 = hogDescriptor.getDescriptor(data: image2, width: 64, height: 64)
            
            XCTAssertEqual(f1, f2, accuracy: 1e-5)
        }
    }
    
    func testDoubleImplAndUInt8ImplEquity() {
        let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
        
        let des = HOGDescriptor(orientations: 9, normalization: .l1)
        let double = des.getDescriptor(data: image1, width: 64, height: 64)
        var uint = [Double](repeating: 0, count: double.count)
        var workspace = des.createWorkspaces(width: 64, height: 64)
        image1.withUnsafeBufferPointer { image1 in
            uint.withUnsafeMutableBufferPointer { uint in
                workspace.double.withUnsafeMutableBufferPointer { ws in
                    des.getDescriptor(data: image1, width: 64, height: 64, descriptor: uint, workspace: ws)
                }
            }
        }
        
        XCTAssertEqual(double, uint, accuracy: 1e-6)
    }
    
    static var allTests = [
        ("testScaleInvariance", testScaleInvariance),
    ]
}
