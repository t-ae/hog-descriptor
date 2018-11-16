#if canImport(Accelerate)
import XCTest
import Accelerate

class VotePerfromanceTests: XCTestCase {
    
    let values = (0..<(2<<18)).map { 5*(sin(Double($0))+1) }
    let numBins = 9
    let iteration = 1000
    
    func testVote_Double() {
        measure {
            for _ in 0..<iteration {
                var bins = [Int](repeating: 0, count: numBins)
                values.withUnsafeBufferPointer {
                    var p = $0.baseAddress!
                    for _ in 0..<$0.count {
                        var index = Int(p.pointee)
                        if index >= numBins {
                            index -= numBins
                        }
                        bins[index] += 1
                        p += 1
                    }
                }
            }
        }
    }
    
    func testVote_UInt8() {
        measure {
            for _ in 0..<iteration {
                var bins = [Int](repeating: 0, count: numBins)
                var indices = [UInt8](repeating: 0, count: values.count)
                vDSP_vfixu8D(values, 1, &indices, 1, UInt(values.count))
                
                indices.withUnsafeBufferPointer {
                    var p = $0.baseAddress!
                    for _ in 0..<$0.count {
                        var index = Int(p.pointee)
                        if index >= numBins {
                            index -= numBins
                        }
                        bins[index] += 1
                        p += 1
                    }
                }
            }
        }
    }
    
    func testVote_Int8() {
        measure {
            for _ in 0..<iteration {
                var bins = [Int](repeating: 0, count: numBins)
                var indices = [Int8](repeating: 0, count: values.count)
                vDSP_vfix8D(values, 1, &indices, 1, UInt(values.count))
                
                indices.withUnsafeBufferPointer {
                    var p = $0.baseAddress!
                    for _ in 0..<$0.count {
                        var index = Int(p.pointee)
                        if index >= numBins {
                            index -= numBins
                        }
                        bins[index] += 1
                        p += 1
                    }
                }
            }
        }
    }
}

#endif
