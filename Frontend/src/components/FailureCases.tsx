import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AlertTriangle, ChevronLeft, ChevronRight, Eye, X } from 'lucide-react';

interface FailureCase {
  id: string;
  image: string;
  predicted: string;
  actual: string;
  confidence: number;
  reason: string;
  description: string;
}

const failureCases: FailureCase[] = [
  {
    id: '1',
    image: 'failure-1',
    predicted: 'Toolbox',
    actual: 'Fire Extinguisher',
    confidence: 0.76,
    reason: 'Partial Occlusion',
    description: 'Fire extinguisher partially hidden behind solar panel array, causing misclassification due to limited visible features.'
  },
  {
    id: '2',
    image: 'failure-2',
    predicted: 'Oxygen Tank',
    actual: 'Toolbox',
    confidence: 0.68,
    reason: 'Similar Shape',
    description: 'Cylindrical toolbox container confused with oxygen tank due to similar geometric profile in low-light conditions.'
  },
  {
    id: '3',
    image: 'failure-3',
    predicted: 'Background',
    actual: 'Fire Extinguisher',
    confidence: 0.23,
    reason: 'Low Contrast',
    description: 'Fire extinguisher blend with module wall due to similar coloring and insufficient lighting contrast.'
  },
  {
    id: '4',
    image: 'failure-4',
    predicted: 'Toolbox',
    actual: 'Oxygen Tank',
    confidence: 0.82,
    reason: 'Unusual Angle',
    description: 'Oxygen tank viewed from unusual perspective during EVA, creating atypical visual signature.'
  },
];

export default function FailureCases() {
  const [selectedCase, setSelectedCase] = useState<FailureCase | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const nextCase = () => {
    setCurrentIndex((prev) => (prev + 1) % failureCases.length);
  };

  const prevCase = () => {
    setCurrentIndex((prev) => (prev - 1 + failureCases.length) % failureCases.length);
  };

  return (
    <section className="py-20 px-4 relative">
      <div className="container mx-auto">
        <motion.div
          className="text-center mb-8 sm:mb-12 px-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="flex flex-col sm:flex-row items-center justify-center mb-4">
            <AlertTriangle className="w-6 h-6 sm:w-8 sm:h-8 text-destructive mb-2 sm:mb-0 sm:mr-3" />
            <h2 className="text-3xl sm:text-4xl font-bold">Failure Case Observatory</h2>
          </div>
          <p className="text-lg sm:text-xl text-muted-foreground">
            Learning from misclassifications to improve orbital detection accuracy
          </p>
        </motion.div>

        {/* Carousel Navigation */}
        <motion.div
          className="flex items-center justify-center mb-6 sm:mb-8 space-x-4 px-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          viewport={{ once: true }}
        >
          <Button variant="outline" size="sm" onClick={prevCase}>
            <ChevronLeft className="w-4 h-4" />
          </Button>
          
          <div className="flex space-x-2">
            {failureCases.map((_, index) => (
              <button
                key={index}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${
                  index === currentIndex 
                    ? 'bg-primary scale-125' 
                    : 'bg-muted hover:bg-primary/50'
                }`}
                onClick={() => setCurrentIndex(index)}
              />
            ))}
          </div>
          
          <Button variant="outline" size="sm" onClick={nextCase}>
            <ChevronRight className="w-4 h-4" />
          </Button>
        </motion.div>

        {/* Main Carousel */}
        <motion.div
          className="relative overflow-hidden px-4"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
        >
          <div 
            className="flex transition-transform duration-500 ease-out"
            style={{ transform: `translateX(-${currentIndex * 100}%)` }}
          >
            {failureCases.map((failureCase, index) => (
              <div key={failureCase.id} className="w-full flex-shrink-0 px-1 sm:px-2">
                <Card className="glass-panel hover-tilt overflow-hidden group cursor-pointer hover:aurora-glow transition-all duration-700">
                  <div className="grid md:grid-cols-2 gap-4 sm:gap-6">
                    {/* Image Section */}
                    <div className="relative aspect-video bg-gradient-to-br from-destructive/20 to-secondary/20 rounded-lg overflow-hidden">
                      {/* Mock failure image with red overlay */}
                      <div className="absolute inset-0 bg-gradient-to-br from-secondary/30 to-card/50" />
                      
                      {/* Red error halo effect */}
                      <motion.div
                        className="absolute inset-0 border-4 border-destructive/50 rounded-lg"
                        animate={{
                          boxShadow: [
                            '0 0 20px hsl(var(--destructive) / 0.3)',
                            '0 0 40px hsl(var(--destructive) / 0.5)',
                            '0 0 20px hsl(var(--destructive) / 0.3)',
                          ],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: "easeInOut",
                        }}
                      />

                      {/* Mock detected object with wrong classification */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <motion.div
                          className="w-24 h-16 border-2 border-destructive rounded-lg bg-destructive/10"
                          animate={{ scale: [1, 1.05, 1] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                        />
                      </div>

                      {/* Error label */}
                      <div className="absolute top-2 left-2 glass rounded-md px-2 py-1 text-xs bg-destructive/20">
                        <span className="text-destructive font-medium">
                          Predicted: {failureCase.predicted}
                        </span>
                      </div>

                      {/* Confidence */}
                      <div className="absolute top-2 right-2 glass rounded-md px-2 py-1 text-xs">
                        <span className="text-muted-foreground">
                          {(failureCase.confidence * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* View details button */}
                      <motion.button
                        className="absolute bottom-2 right-2 p-2 glass rounded-full hover:bg-primary/20 transition-colors"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => setSelectedCase(failureCase)}
                      >
                        <Eye className="w-4 h-4 text-primary" />
                      </motion.button>
                    </div>

                    {/* Details Section */}
                    <div className="space-y-3 sm:space-y-4 p-3 sm:p-4">
                      <div className="space-y-2">
                        <h3 className="text-base sm:text-lg font-semibold">Misclassification Analysis</h3>
                        <p className="text-xs sm:text-sm text-muted-foreground">
                          {failureCase.description}
                        </p>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Predicted:</span>
                          <Badge variant="destructive">{failureCase.predicted}</Badge>
                        </div>
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Actual:</span>
                          <Badge variant="default">{failureCase.actual}</Badge>
                        </div>
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Reason:</span>
                          <Badge variant="outline">{failureCase.reason}</Badge>
                        </div>
                      </div>

                      <div className="pt-4 border-t border-border/50">
                        <h4 className="text-sm font-medium mb-2">Improvement Strategy</h4>
                        <p className="text-xs text-muted-foreground">
                          Enhance training with {failureCase.reason.toLowerCase()} scenarios using Falcon's synthetic data generation.
                        </p>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Summary Stats */}
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6 mt-8 sm:mt-12 px-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          viewport={{ once: true }}
        >
          <Card className="glass-panel hover-tilt text-center hover:cosmic-glow transition-all duration-300">
            <div className="text-2xl sm:text-3xl font-bold text-destructive mb-2 counter">5.3%</div>
            <div className="text-xs sm:text-sm text-muted-foreground">False Positive Rate</div>
          </Card>
          
          <Card className="glass-panel hover-tilt text-center hover:aurora-glow transition-all duration-300">
            <div className="text-2xl sm:text-3xl font-bold text-accent mb-2 counter">12</div>
            <div className="text-xs sm:text-sm text-muted-foreground">Common Failure Patterns</div>
          </Card>
          
          <Card className="glass-panel hover-tilt text-center hover:cosmic-glow transition-all duration-300">
            <div className="text-2xl sm:text-3xl font-bold text-primary mb-2 counter">87%</div>
            <div className="text-xs sm:text-sm text-muted-foreground">Improvement Since v1.0</div>
          </Card>
        </motion.div>
      </div>

      {/* Detailed Modal */}
      <AnimatePresence>
        {selectedCase && (
          <motion.div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedCase(null)}
          >
            <motion.div
              className="glass-card max-w-2xl w-full max-h-[80vh] overflow-y-auto"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-semibold">Failure Case Analysis</h3>
                <Button variant="ghost" size="sm" onClick={() => setSelectedCase(null)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-6">
                <div className="aspect-video bg-gradient-to-br from-destructive/20 to-secondary/20 rounded-lg" />
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Classification Error</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Predicted:</span>
                        <Badge variant="destructive">{selectedCase.predicted}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Actual:</span>
                        <Badge variant="default">{selectedCase.actual}</Badge>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Technical Details</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Confidence:</span>
                        <span className="text-sm">{(selectedCase.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Failure Type:</span>
                        <Badge variant="outline">{selectedCase.reason}</Badge>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Analysis & Solution</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {selectedCase.description} This case will be addressed in the next training iteration by incorporating additional synthetic data variations generated through Falcon's digital twin environment.
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}