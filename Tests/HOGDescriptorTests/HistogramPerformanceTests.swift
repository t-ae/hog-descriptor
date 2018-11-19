import XCTest
import Accelerate

class HistogramPerformanceTests: XCTestCase {
    
    let width = 512
    let height = 512
    var image: [UInt8]!
    
    let orientations = 9
    
    override func setUp() {
        image = (0..<width*height).map { _ in UInt8.random(in: 0...255) }
    }
    
    func calculateHistogram_Accelerate() -> [Double] {
        var double = [Double](repeating: 0, count: image.count)
        var gx = double
        var gy = double
        var gradD = double
        var magnitudes = double
        vDSP_vfltu8D(image, 1, &double, 1, UInt(double.count))
        
        gx.withUnsafeMutableBufferPointer { gx in
            double.withUnsafeBufferPointer { double in
                var src0 = double.baseAddress!
                var src1 = double.baseAddress! + 2
                var dst = gx.baseAddress! + 1
                for _ in 0..<height {
                    vDSP_vsubD(src0, 1, src1, 1, dst, 1, UInt(width-2))
                    src0 += width
                    src1 += width
                    dst += width
                }
            }
        }
        gy.withUnsafeMutableBufferPointer { gy in
            double.withUnsafeBufferPointer { double in
                let src0 = double.baseAddress!
                let src1 = double.baseAddress! + (2*width)
                let dst = gy.baseAddress! + width
                
                vDSP_vsubD(src0, 1, src1, 1, dst, 1, UInt(width*(height-2)))
            }
        }
        
        var grad = [UInt8](repeating: 0, count: gradD.count)
        var cnt = Int32(double.count)
        vvatan2(&gradD, gy, gx, &cnt)
        var multiplier = Double(orientations) / .pi
        var adder = Double(orientations)
        vDSP_vsmsaD(gradD, 1,
                    &multiplier, &adder,
                    &gradD, 1,
                    UInt(gradD.count)) // [0, 2*orientation]
        vDSP_vfixu8D(gradD, 1, &grad, 1, UInt(grad.count))
        
        vDSP_vdistD(gy, 1, gx, 1, &magnitudes, 1, UInt(double.count))
        
        var histogram = [Double](repeating: 0, count: orientations)
        for y in 0..<height {
            for x in 0..<width {
                var bin = Int(grad[y*width+x])
                while bin >= orientations {
                    bin -= orientations
                }
                histogram[bin] += magnitudes[y*width+x]
            }
        }
        return histogram
    }
    
    func testAccelerate() {
        
        let answer = calculateHistogram_Accelerate()
        
        measure {
            for _ in 0..<100 {
                let result = calculateHistogram_Accelerate()
                XCTAssertEqual(result, answer)
            }
        }
    }
    
    func testNoAccelerate() {
        func calculateHistogram() -> [Double] {
            var double = [Double](repeating: 0, count: image.count)
            vDSP_vfltu8D(image, 1, &double, 1, UInt(double.count))
            
            var histogram = [Double](repeating: 0, count: orientations)
            for y in 0..<height {
                for x in 0..<width {
                    let gx: Double
                    if x == 0 || x == width-1 {
                        gx = 0
                    } else {
                        gx = double[y*width+x+1] - double[y*width+x-1]
                    }
                    let gy: Double
                    if y == 0 || y == height-1 {
                        gy = 0
                    } else {
                        gy = double[(y+1)*width+x] - double[(y-1)*width+x]
                    }
                    let angle = atan2(gy, gx)
                    var bin = Int((angle / .pi + 1) * Double(orientations))
                    while bin >= orientations {
                        bin -= orientations
                    }
                    let magnitude = hypot(gy, gx)
                    histogram[bin] += magnitude
                }
            }
            return histogram
        }
        
        let answer = calculateHistogram_Accelerate()
        
        measure {
            for _ in 0..<100 {
                let result = calculateHistogram()
                XCTAssertEqual(result, answer)
            }
        }
    }

    func testLookupTable() {
        
        var table = [Int](repeating: 0, count: 512*512)
        for y in -255...255 {
            let yy = y + 255
            for x in -255...255 {
                let xx = x + 255
                
                let angle = atan2(Double(y), Double(x))
                
                var bin = Int((angle / .pi + 1) * Double(orientations))
                while bin >= orientations {
                    bin -= orientations
                }
                
                table[yy<<9 | xx] = bin
            }
        }
        
        func calculateHistogram() -> [Double] {
            var histogram = [Double](repeating: 0, count: orientations)
            for y in 0..<height {
                for x in 0..<width {
                    let gx: Int
                    if x == 0 || x == width-1 {
                        gx = 0
                    } else {
                        gx = Int(image[y*width+x+1]) - Int(image[y*width+x-1])
                    }
                    let gy: Int
                    if y == 0 || y == height-1 {
                        gy = 0
                    } else {
                        gy = Int(image[(y+1)*width+x]) - Int(image[(y-1)*width+x])
                    }
                    let bin = table[(gy+255)<<9 | (gx+255)]
                    let magnitude = hypot(Double(gy), Double(gx))
                    histogram[bin] += magnitude
                }
            }
            return histogram
        }
        
        let answer = calculateHistogram_Accelerate()
        
        measure {
            for _ in 0..<100 {
                let result = calculateHistogram()
                XCTAssertEqual(result, answer)
            }
        }
    }
}
