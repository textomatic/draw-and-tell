//
//  UIImage+Extension.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import Foundation
import UIKit

extension UIImage {
    
    func resizeTo(size: CGSize) -> UIImage? {
        autoreleasepool { () -> UIImage? in
            UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
            
            self.draw(in: CGRect(origin: CGPoint.zero, size: size))
            let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
            
            UIGraphicsEndImageContext()
            
            return resizedImage
        }
    }
    
    func toBuffer() -> CVPixelBuffer? {
        autoreleasepool { () -> CVPixelBuffer? in
            let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
            var pixelBuffer : CVPixelBuffer?
            let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(self.size.width), Int(self.size.height), kCVPixelFormatType_OneComponent8, attrs, &pixelBuffer)
            
            guard (status == kCVReturnSuccess) else {
                return nil
            }
            
            CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
            let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
            let grayColorSpace = CGColorSpaceCreateDeviceGray()
            let context = CGContext(data: pixelData, width: Int(self.size.width), height: Int(self.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: grayColorSpace, bitmapInfo: 0)
            
            context?.translateBy(x: 0, y: self.size.height)
            context?.scaleBy(x: 1.0, y: -1.0)
            
            UIGraphicsPushContext(context!)
            self.draw(in: CGRect(x: 0, y: 0, width: self.size.width, height: self.size.height))
            UIGraphicsPopContext()
            CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            
            return pixelBuffer
        }
    }
}
