//
//  CanvasView.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import SwiftUI

struct CanvasView: View {
    // MARK: - Properties
    @Binding var currentLine: Line
    @Binding var lines: [Line]
    
    // MARK: - Body
    var body: some View {
        
        VStack {
            
            // Canvas for doodling
            DoodleView(
                currentLine: currentLine,
                lines: lines
            )
            .gesture(
                DragGesture(
                    minimumDistance: 0,
                    coordinateSpace: .local
                )
                .onChanged { value in
                    currentLine.points.append(value.location)
                    lines.append(currentLine)
                }
                .onEnded { value in
                    self.lines.append(currentLine)
                    self.currentLine = Line()
                }
            ) //: DragGesture
            
        } //: VStack
        
    } //: Body
    
}

struct CanvasView_Previews: PreviewProvider {
    static var previews: some View {
        // Light Theme
        CanvasView(
            currentLine: .constant(Line()),
            lines: .constant([Line]())
        )
        .preferredColorScheme(.light)
        
        // Dark Theme
        CanvasView(
            currentLine: .constant(Line()),
            lines: .constant([Line]())
        )
        .preferredColorScheme(.dark)
    }
}
