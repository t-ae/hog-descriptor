import XCTest
@testable import HOGFeatureExtractor

final class HOGFeatureExtractorTests: XCTestCase {
    func testScaleInvariance() {
        do {
            let extractor = HOGFeatureExtractor(cellSpan: 3, blockSpan: 4, orientation: 9, normalization: .l1)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.extract(data: image1, width: 64, height: 64)
            let f2 = extractor.extract(data: image2, width: 64, height: 64)
            
            for (c1, c2) in zip(f1, f2) {
                XCTAssertEqual(c1, c2, accuracy: 1e-8)
            }
        }
        do {
            let extractor = HOGFeatureExtractor(cellSpan: 8, blockSpan: 4, orientation: 5, normalization: .l2)
            
            let image1 = (0..<64*64).map { _ in UInt8.random(in: 0...255) }
            let image2 = image1.map { Double($0)/255 }
            
            let f1 = extractor.extract(data: image1, width: 64, height: 64)
            let f2 = extractor.extract(data: image2, width: 64, height: 64)
            
            for (c1, c2) in zip(f1, f2) {
                XCTAssertEqual(c1, c2, accuracy: 1e-5)
            }
        }
    }

    static var allTests = [
        ("testScaleInvariance", testScaleInvariance),
    ]
}
