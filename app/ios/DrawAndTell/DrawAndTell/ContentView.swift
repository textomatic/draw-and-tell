//
//  ContentView.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import AVFoundation
import SwiftUI

struct ContentView: View {
    // MARK: - Properties
    @Environment(\.displayScale) var displayScale
    @StateObject private var viewModel = DoodleRecognitionViewModel()
    // Instantite speech synthesizer as state variable to support text-to-speech of animal fact
    @State var synthesizer = AVSpeechSynthesizer()
    
    // MARK: - Body
    var body: some View {
        
        GeometryReader { geometry in
            
            VStack {
                
                // Text for doodle prompt
                Text(viewModel.drawPrompt)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
                    .padding()

                Spacer()
                
                // Text for doodle label
                Text(viewModel.doodleClass)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(Color.green)
                    .padding(10)
                
                Spacer()
                
                // Text for animal fact
                Text(viewModel.animalFact)
                    .font(.title3)
                    .multilineTextAlignment(.center)
                    .italic()
                    .foregroundColor(Color.brown)
                    .padding(.horizontal, 15)
                
                // Button for text-to-speech of animal fact
                if !viewModel.animalFact.isEmpty {
                    Button() {
                        let utterance = AVSpeechUtterance(string: viewModel.animalFact)
                        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                        synthesizer.speak(utterance)
                    } label: {
                        Label("Listen", systemImage: "speaker")
                        //Image(systemName: "speaker")
                            .font(.headline)
                            .fontWeight(.medium)
                    }
                    .padding(1)
                }
                
                Spacer()
                
                // View of Canvas for doodling
                CanvasView(
                    currentLine: $viewModel.currentLine,
                    lines: $viewModel.lines
                )
                .frame(maxHeight: 500)
                .background(Color.white)
                .border(.secondary, width: 0.5)
                .padding(.vertical, 20)
                
                Spacer()
                
                HStack(alignment: .center, spacing: 30) {
                    
                    // Button for triggering doodle recognition
                    Button {
                        withAnimation(.linear(duration: 0.2)) {
                            viewModel.classifyDoodle(displayScale, size: geometry.size)
                        }
                    } label: {
                        Text("Guess")
                            .fontWeight(.bold)
                            .font(.system(.title, design: .rounded))
                    }
                    .padding()
                    .frame(width: 150)
                    .foregroundColor(.white)
                    .background(Color.orange)
                    .cornerRadius(20)
                    .disabled($viewModel.lines.isEmpty)
                    
                    // Button for clearing out the canvas
                    Button {
                        withAnimation(.linear(duration: 0.2)) {
                            viewModel.clearDoodle()
                        }
                    } label: {
                        Text("Clear")
                            .fontWeight(.bold)
                            .font(.system(.title, design: .rounded))
                    }
                    .padding()
                    .frame(width: 150)
                    .foregroundColor(.white)
                    .background(Color.accentColor)
                    .cornerRadius(20)
                
                } //: HStack
                
            } //: VStack
            
        } //: GeometryReader
        
    } //: Body
    
} //: ContentView

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
