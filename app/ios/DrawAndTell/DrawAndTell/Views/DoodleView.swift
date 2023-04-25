//
//  DoodleView.swift
//  DrawAndTell
//
//  Created by Shen Juin Lee on 4/23/23.
//

import SwiftUI

struct DoodleView: View {
    // MARK: - Properties
    let currentLine: Line
    let lines: [Line]
    
    // MARK: - Body
    var body: some View {
        
        VStack {
            
            // Canvas view type supports immediate mode drawing operations
            Canvas { context, size in
                
                for line in lines {
                    var path = Path()
                    path.addLines(line.points)
                    
                    context.stroke(
                        path,
                        with: .color(line.color),
                        lineWidth: line.lineWidth
                    )
                }

            } //: Canvas
            
        } //: VStack
        
    } //: Body
    
}
