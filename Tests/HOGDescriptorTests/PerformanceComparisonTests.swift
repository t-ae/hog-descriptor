//import XCTest
//@testable import HOGDescriptor
//import Accelerate
//
//class PerformanceComparisonTests: XCTestCase {
//    
//    let hogDescriptor = HOGDescriptor()
//    var image: [UInt8]!
//    let width = 512
//    let height = 512
//    let orientations = 9
//    let pixelsPerCell = (x: 8, y: 8)
//    
//    override func setUp() {
//        image = (0..<width*height).map { _ in UInt8.random(in: 0...255) }
//    }
//    
//    func calculateHistogramAccelerate() -> [Double]{
//        let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
//        
//        var doubleImage = [Double](repeating: 0, count: image.count)
//        vDSP_vfltu8D(image, 1, &doubleImage, 1, UInt(image.count))
//        let (gradX, gradY) = hogDescriptor.derivate(data: doubleImage, width: width, height: height)
//        
//        // calculate gradient directions and magnitudes
//        var grad = [Double](repeating: 0, count: gradX.count)
//        do {
//            var _cnt = Int32(grad.count)
//            vvatan2(&grad, gradY, gradX, &_cnt) // [-pi, pi]
//            var multiplier = Double(orientations) / .pi
//            var adder = Double(orientations)
//            vDSP_vsmsaD(grad, 1, &multiplier, &adder, &grad, 1, UInt(grad.count)) // [0, 2*orientation]
//        }
//        
//        var magnitude = [Double](repeating: 0, count: gradX.count)
//        vDSP_vdistD(gradX, 1, gradY, 1, &magnitude, 1, UInt(magnitude.count))
//        
//        // accumulate to histograms
//        
//        // N-D array of [numberOfCells.y, numberOfCells.x, orientations]
//        var histograms = [Double](repeating: 0, count: numberOfCells.y*numberOfCells.x*orientations)
//        
//        for cellY in 0..<numberOfCells.y {
//            for cellX in 0..<numberOfCells.x {
//                let histogramIndex = (cellY * numberOfCells.x + cellX) * orientations
//                for y in cellY*pixelsPerCell.y..<(cellY+1)*pixelsPerCell.y {
//                    for x in cellX*pixelsPerCell.x..<(cellX+1)*pixelsPerCell.x {
//                        var directionIndex = Int(grad[y*width+x])
//                        while directionIndex >= orientations {
//                            directionIndex -= orientations
//                        }
//                        histograms[histogramIndex + directionIndex] += magnitude[y*width+x]
//                    }
//                }
//            }
//        }
//        
//        return histograms
//    }
//
//    func testCalcHistogramSeparate() throws {
//        
//        let answer = calculateHistogramAccelerate()
//        
//        measure {
//            for _ in 0..<100 {
//                let histograms = calculateHistogramAccelerate()
////                XCTAssertEqual(histograms, answer)
//            }
//        }
//    }
//    
//    func testCalcHistogramFuse() {
//        func calculateHistogram() -> [Double] {
//            let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
//            
//            var doubleImage = [Double](repeating: 0, count: image.count)
//            vDSP_vfltu8D(image, 1, &doubleImage, 1, UInt(image.count))
//            
//            var histograms = [Double](repeating: 0, count: numberOfCells.y*numberOfCells.x*orientations)
//            
//            for cellY in 0..<numberOfCells.y {
//                for cellX in 0..<numberOfCells.x {
//                    let histogramIndex = (cellY * numberOfCells.x + cellX) * orientations
//                    for y in cellY*pixelsPerCell.y..<(cellY+1)*pixelsPerCell.y {
//                        for x in cellX*pixelsPerCell.x..<(cellX+1)*pixelsPerCell.x {
//                            let index = y*width+x
//                            let gradY = (y == 0 || y == height-1) ? 0
//                                : doubleImage[index+width] - doubleImage[index-width]
//                            let gradX = (x == 0 || x == width-1) ? 0
//                                : doubleImage[index+1] - doubleImage[index-1]
//                            
//                            let grad = atan2(gradY, gradX) // [-pi, pi]
//                            let magnitude = hypot(gradY, gradX)
//                            
//                            var directionIndex = Int((grad / .pi + 1) * Double(orientations))
//                            while directionIndex >= orientations {
//                                directionIndex -= orientations
//                            }
//                            histograms[histogramIndex + directionIndex] += magnitude
//                        }
//                    }
//                }
//            }
//            return histograms
//        }
//        
//        let answer = calculateHistogramAccelerate()
//        
//        measure {
//            for _ in 0..<100 {
//                let histograms = calculateHistogram()
////                XCTAssertEqual(histograms, answer)
//            }
//        }
//    }
//    
//    func testCalcHistogramLookupTable() {
//        var table = [Double](repeating: 0, count: 512*512)
//        for dy in -255...255 {
//            for dx in -255...255 {
//                table[(dy+255)<<9 | (dx + 255)] = atan2(Double(dy), Double(dx))
//            }
//        }
//        
//        func calculateHistogram() -> [Double] {
//            let numberOfCells = (x: width / pixelsPerCell.x, y: height / pixelsPerCell.y)
//            
//            var histograms = [Double](repeating: 0, count: numberOfCells.y*numberOfCells.x*orientations)
//            
//            for cellY in 0..<numberOfCells.y {
//                for cellX in 0..<numberOfCells.x {
//                    let histogramIndex = (cellY * numberOfCells.x + cellX) * orientations
//                    for y in cellY*pixelsPerCell.y..<(cellY+1)*pixelsPerCell.y {
//                        for x in cellX*pixelsPerCell.x..<(cellX+1)*pixelsPerCell.x {
//                            let index = y*width+x
//                            let gradY = (y == 0 || y == height-1) ? 0
//                                : Int(image[index+width]) - Int(image[index-width])
//                            let gradX = (x == 0 || x == width-1) ? 0
//                                : Int(image[index+1]) - Int(image[index-1])
//                            
//                            let grad = table[(gradY+255)<<9 | (gradX+255)]
//                            let magnitude = hypot(Double(gradY), Double(gradX))
//                            
//                            var directionIndex = Int((grad / .pi + 1) * Double(orientations))
//                            while directionIndex >= orientations {
//                                directionIndex -= orientations
//                            }
//                            histograms[histogramIndex + directionIndex] += magnitude
//                        }
//                    }
//                }
//            }
//            return histograms
//        }
//        
//        let answer = calculateHistogramAccelerate()
//        
//        measure {
//            for _ in 0..<100 {
//                let histograms = calculateHistogram()
////                XCTAssertEqual(histograms, answer)
//            }
//        }
//    }
//}
