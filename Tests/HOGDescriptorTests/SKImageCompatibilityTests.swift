import XCTest
@testable import HOGDescriptor

class SKImageCompatibilityTests: XCTestCase {
    
    func skimageEquivalent(imageSize: (Int, Int),
                           orientations: Int,
                           pixelsPerCell: (Int, Int),
                           cellsPerBlock: (Int, Int),
                           normalization: HOGDescriptor.NormalizationMethod) -> [Double] {
        
        let image = (0..<imageSize.0*imageSize.1).map { abs(sin(Double($0))) }
        
        let extractor = HOGDescriptor(orientations: orientations,
                                      pixelsPerCell: pixelsPerCell,
                                      cellsPerBlock: cellsPerBlock,
                                      normalization: normalization)
        
        return extractor.getDescriptor(data: image, width: imageSize.0, height: imageSize.1)
    }
    
    let eps = 1e-4

    func testA() {
        let f = skimageEquivalent(imageSize: (4, 4),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_4_ori_9_ppc_2_2_bpc_2_2_L1, accuracy: eps)
    }
    
    func testB() {
        let f = skimageEquivalent(imageSize: (4, 5),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_5_ori_9_ppc_2_2_bpc_2_2_L1, accuracy: eps)
    }
    
    func testC() {
        let f = skimageEquivalent(imageSize: (4, 6),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_6_ori_9_ppc_2_2_bpc_2_2_L1, accuracy: eps)
    }
    
    func testD() {
        let f = skimageEquivalent(imageSize: (6, 4),
                                  orientations: 3,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (2, 2),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_6_4_ori_3_ppc_2_2_bpc_2_2_L1, accuracy: eps)
    }
    
    func testE() {
        let f = skimageEquivalent(imageSize: (6, 6),
                                  orientations: 9,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l2)
        
        XCTAssertEqual(f, size_6_6_ori_9_ppc_2_2_bpc_3_3_L2, accuracy: eps)
    }
    
    func testF() {
        let f = skimageEquivalent(imageSize: (6, 6),
                                  orientations: 5,
                                  pixelsPerCell: (2, 2),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_6_6_ori_5_ppc_2_2_bpc_3_3_L1, accuracy: eps)
    }
    
    func testG() {
        let f = skimageEquivalent(imageSize: (4, 4),
                                  orientations: 5,
                                  pixelsPerCell: (4, 4),
                                  cellsPerBlock: (1, 1),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_4_4_ori_5_ppc_4_4_bpc_1_1_L1, accuracy: eps)
    }
    
    func testH() {
        let f = skimageEquivalent(imageSize: (16, 16),
                                  orientations: 3,
                                  pixelsPerCell: (4, 4),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l1)
        
        XCTAssertEqual(f, size_16_16_ori_3_ppc_4_4_bpc_3_3_L1, accuracy: eps)
    }
    
    func testI() {
        let f = skimageEquivalent(imageSize: (16, 16),
                                  orientations: 9,
                                  pixelsPerCell: (4, 4),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l2)
        
        XCTAssertEqual(f, size_16_16_ori_9_ppc_4_4_bpc_3_3_L2, accuracy: eps)
    }
    
    func testJ() {
        let f = skimageEquivalent(imageSize: (16, 16),
                                  orientations: 9,
                                  pixelsPerCell: (4, 4),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l1sqrt)
        
        XCTAssertEqual(f, size_16_16_ori_9_ppc_4_4_bpc_3_3_L1_sqrt, accuracy: eps)
    }
    
    func testK() {
        let f = skimageEquivalent(imageSize: (16, 16),
                                  orientations: 9,
                                  pixelsPerCell: (4, 4),
                                  cellsPerBlock: (3, 3),
                                  normalization: .l2Hys)
        
        XCTAssertEqual(f, size_16_16_ori_9_ppc_4_4_bpc_3_3_L2_Hys, accuracy: eps)
    }
}
