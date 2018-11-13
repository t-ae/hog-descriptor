import XCTest
@testable import HOGDescriptor
import Accelerate

class GradientCalculationTests: XCTestCase {
    
    let hogDescriptor = HOGDescriptor()
    let width = 512
    let height = 512

    func testCalc() {
        let image = (0..<width*height).map { _ in UInt8.random(in: 0...255) }
        
        measure {
            for _ in 0..<100 {
                var doubleImage = [Double](repeating: 0, count: image.count)
                vDSP_vfltu8D(image, 1, &doubleImage, 1, UInt(image.count))
                let (gradX, gradY) = hogDescriptor.derivate(data: doubleImage, width: width, height: height)
                var grad = [Double](repeating: 0, count: width*height)
                var count = Int32(width*height)
                vvatan2(&grad, gradY, gradX, &count)
            }
        }
    }
    
    func testCalcWithLookupTable() {
        let image = (0..<width*height).map { _ in UInt8.random(in: 0...255) }
        
        var table = [Double](repeating: 0, count: 511*511)
        for dy in -255...255 {
            for dx in -255...255 {
                table[(dy+255)*511 + (dx + 255)] = atan2(Double(dy), Double(dx))
            }
        }
        
        measure {
            for _ in 0..<100 {
                let intImage = image.map { Int($0) }
                var gradX = [Int](repeating: 0, count: width*height)
                intImage.withUnsafeBufferPointer {
                    var src1 = $0.baseAddress!
                    var src2 = $0.baseAddress! + 2
                    gradX.withUnsafeMutableBufferPointer {
                        var dst = $0.baseAddress! + 1
                        for _ in 0..<height {
                            for _ in 1..<width-1 {
                                dst.pointee = src2.pointee - src1.pointee
                                dst += 1
                                src1 += 1
                                src2 += 1
                            }
                            dst += 2
                            src1 += 2
                            src2 += 2
                        }
                    }
                }
                var gradY = [Int](repeating: 0, count: width*height)
                intImage.withUnsafeBufferPointer {
                    var src1 = $0.baseAddress!
                    var src2 = $0.baseAddress! + 2*width
                    gradY.withUnsafeMutableBufferPointer {
                        var dst = $0.baseAddress! + width
                        for _ in 0..<width*(height-2) {
                            dst.pointee = src2.pointee - src1.pointee
                            dst += 1
                            src1 += 1
                            src2 += 1
                        }
                    }
                }
                
                var grad = [Double](repeating: 0, count: width*height)
                
                gradY.withUnsafeBufferPointer {
                    var srcY = $0.baseAddress!
                    gradX.withUnsafeBufferPointer {
                        var srcX = $0.baseAddress!
                        grad.withUnsafeMutableBufferPointer {
                            var dst = $0.baseAddress!
                            for _ in 0..<width*height {
                                let index = (srcY.pointee+255)*511 + (srcX.pointee+255)
                                dst.pointee = table[index]
                                dst += 1
                                srcY += 1
                                srcX += 1
                            }
                        }
                    }
                }
            }
        }
    }
}
