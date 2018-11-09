import XCTest
@testable import HOGFeatureExtractor

class SKImageCompatibilityTests: XCTestCase {
    
    func skimageEquivalent(imageSize: (Int, Int),
                           orientations: Int,
                           pixelsPerCell: (Int, Int),
                           cellsPerBlock: (Int, Int),
                           normalization: HOGFeatureExtractor.NormalizationMethod) -> [Double] {
        
        let image = (0..<imageSize.0*imageSize.1).map { abs(sin(Double($0))) }
        
        let extractor = HOGFeatureExtractor(pixelsInCell: pixelsPerCell,
                                            cellsInBlock: cellsPerBlock,
                                            orientation: orientations,
                                            normalization: normalization)
        
        return extractor.extract(data: image, width: imageSize.0, height: imageSize.1)
    }

    func testA() {
        let f = skimageEquivalent(imageSize: (4, 4),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_4_ori_9_ppc_2_2_bpc_2_2_l1, accuracy: 1e-4)
    }
    
    func testB() {
        let f = skimageEquivalent(imageSize: (4, 5),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_5_ori_9_ppc_2_2_bpc_2_2_l1, accuracy: 1e-4)
    }
    
    func testC() {
        let f = skimageEquivalent(imageSize: (4, 6),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_6_ori_9_ppc_2_2_bpc_2_2_l1, accuracy: 1e-4)
    }
}
