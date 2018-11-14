import Foundation

func getTestResourceRootDir() -> URL {
    let current = #file
    
    var url = URL(fileURLWithPath: current)
    
    while url.lastPathComponent != "Tests" {
        url.deleteLastPathComponent()
    }
    
    return url.appendingPathComponent("TestResources")
}

func loadText(url: URL) throws -> [Double] {
    let str = try String(contentsOf: url)
    let values = str.components(separatedBy: CharacterSet.whitespacesAndNewlines)
    
    return values.compactMap { Double($0) }
}

func loadAstronautGray() throws -> [Double] {
    // astro = skimage.data.astronaut()
    // gray = skimage.color.rgb2gray(astro)
    // np.savetxt("astronaut_gray.txt", gray)
    let url = getTestResourceRootDir().appendingPathComponent("astronaut_gray.txt")
    return try loadText(url: url)
}

func loadAstronautGradX() throws -> [Double] {
    // astro = skimage.data.astronaut()
    // gray = skimage.color.rgb2gray(astro)
    // gradX = np.zeros_like(gray)
    // gradX[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    // np.savetxt("astronaut_grad_x.txt", gradX)
    let url = getTestResourceRootDir().appendingPathComponent("astronaut_grad_x.txt")
    return try loadText(url: url)
}

func loadAstronautGradY() throws -> [Double] {
    // astro = skimage.data.astronaut()
    // gray = skimage.color.rgb2gray(astro)
    // gradY = np.zeros_like(gray)
    // gradY[1:-1, :] = gray[2:, :] - gray[:-2, :]
    // np.savetxt("astronaut_grad_y.txt", gradY)
    let url = getTestResourceRootDir().appendingPathComponent("astronaut_grad_y.txt")
    return try loadText(url: url)
}

func loadAstronautHOG_L1() throws -> [Double] {
    // astro_l1 = np.load("/{SKIMAGEROOT}/skimage/data/astronaut_GRAY_hog_L1.npy")
    // np.savetxt("astronaut_hog_l1", astro_l1)
    let url = getTestResourceRootDir().appendingPathComponent("astronaut_hog_l1.txt")
    return try loadText(url: url)
}

func loadAstronautHOG_L2Hys() throws -> [Double] {
    // astro_l2_hys = np.load("/{SKIMAGEROOT}/skimage/data/astronaut_GRAY_hog_L2-Hys.npy")
    // np.savetxt("astronaut_hog_l1", astro_l2_hys)
    let url = getTestResourceRootDir().appendingPathComponent("astronaut_hog_l2_hys.txt")
    return try loadText(url: url)
}
