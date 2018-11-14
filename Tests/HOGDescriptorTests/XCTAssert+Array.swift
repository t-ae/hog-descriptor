import XCTest

func XCTAssertEqual(_ array1: [Double],
                    _ array2: [Double],
                    accuracy: Double,
                    file: StaticString = #file,
                    line: UInt = #line) {
    XCTAssertEqual(array1.count, array2.count, file: file, line: line)
    
    for (i, (e1, e2)) in zip(array1, array2).enumerated() {
        XCTAssertEqual(e1, e2, accuracy: accuracy, "Index \(i)", file: file, line: line)
    }
}
