//
//  DoodleRecognizerService.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import Foundation
import CoreML
import UIKit

/**
Service for performing doodle recognition. Initialized with a custom-trained EfficientNet B3 machine learning model.
 */
class DoodleRecognitionService {
    // MARK: - Properties
    private var model: drawandtell_v1? = nil
    
    // MARK: - Initializer
    init() {
        self.model = try? drawandtell_v1(configuration: MLModelConfiguration())
    }
    
    // MARK: Deinit
    deinit {
        self.model = nil
    }
    
    // MARK: - Functions
    /**
        Classifies doodle and returns its label.
     
     - Parameter image: UIImage of the doodle
     - Returns: String of the doodle's class label
     */
    func classifyDoodle(_ image: UIImage) -> String? {
        guard let model = model,
              let bufferImage = image.toBuffer() else {
            return nil
        }
        
        guard let output = try? model.prediction(image_input: bufferImage) else {
            return nil
        }
        
        return output.classLabel
    }
}
