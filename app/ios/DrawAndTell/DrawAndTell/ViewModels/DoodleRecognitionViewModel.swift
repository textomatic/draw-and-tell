//
//  DoodleRecognizerViewModel.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import Foundation
import SwiftUI

/**
 Main view model of the application. Contains logic to handle doodle recognition and allows CanvasView to observe changes of state.
 */
@MainActor
class DoodleRecognitionViewModel: ObservableObject {
    // MARK: - Properties
    @Published var drawPrompt: String = "Draw an animal and let us guess what it is!"
    @Published var doodleClass: String = ""
    @Published var currentLine = Line()
    @Published var lines: [Line] = []
    @Published var animalFact: String = ""
    @Published var animalFacts: Dictionary <String, Any>
    
    private var doodleRecognitionService: DoodleRecognitionService? = nil
    
    // MARK: - Initializer
    init() {
        self.doodleRecognitionService = DoodleRecognitionService()
        self.animalFacts = DoodleRecognitionViewModel.readJson()!
    }
    
    // MARK: - Functions
    /**
    Reads and decodes the `animal_facts_dict` JSON file and returns it as a dictionary.
     
     - Parameters: Nil
     - Returns: Dictionary <String, Any>
     */
    class func readJson() -> [String: Any]? {
        // Get url for file
        guard let filePath = Bundle.main.path(forResource: "animal_facts_dict", ofType: "json") else {
            print("JSON file could not be located at the given file name")
            return nil
        }

        do {
            // Get animal facts data from JSON file
            let fileUrl = URL(fileURLWithPath: filePath)
            let data = try Data(contentsOf: fileUrl)

            // Decode data to a Dictionary<String, Any> object
            guard let dictionary = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
                print("Could not cast JSON content as a Dictionary<String, Any>")
                return nil
            }

            // Return animal facts dictionary
            return dictionary
            
        } catch {
            // Print error if something went wrong
            print("Error: \(error)")
            return nil
        }
    }
    
    /**
    Calls DoodleRecognitionService to classify the doodle drawn by user and updates view model with label of doodle and a fact related to the label.
     
     - Parameters:
        - displayScale: Display scale of the environment
        - size: Size of the GeometryReader container view
     
     - Returns: Nil
     */
    func classifyDoodle(_ displayScale: CGFloat, size: CGSize) {
        // Instantiate doodleRecognizerService as variable
        guard let doodleRecognitionService = doodleRecognitionService else {
            return
        }
        
        // Create variable for UI ImageRenderer so that doodle can be rendered as a UI Image
        let renderer = ImageRenderer(
            content: DoodleView(
                currentLine: self.currentLine,
                lines: self.lines
            )
            .frame(width: size.width, height: 500)
        )
        renderer.scale = displayScale
        
        guard let uiImage = renderer.uiImage,
              let data = uiImage.jpegData(compressionQuality: 1.0),
              let fullImage = UIImage(data: data),
              let resizedImage = fullImage.resizeTo(size: CGSize(width: 224, height: 224)) else {
            return
        }
        
        // Save doodle to Photos album so that user can refer to it later
        UIImageWriteToSavedPhotosAlbum(fullImage, nil, nil, nil)
        
        // Pass doodle image to doodleRecognizerService for prediction
        guard let doodleLabel = doodleRecognitionService.classifyDoodle(resizedImage) else {
            return
        }
        
        // Assign doodle prediction to variable
        self.doodleClass = doodleLabel
        // Obtain array of animal facts related to label of doodle
        let facts: [String] = self.animalFacts[doodleLabel] as! [String]
        // Get a random fact from the array and assign it to variable
        let fact = "Fun fact: " + facts.randomElement()! as String
        self.animalFact = fact
    }
    
    /**
    Clears the canvas of existing doodle.
     
     - Parameters: Nil
     - Returns: Nil
     */
    func clearDoodle() {
        self.currentLine = Line()
        self.lines = []
        self.doodleClass = ""
        self.animalFact = ""
    }
}
