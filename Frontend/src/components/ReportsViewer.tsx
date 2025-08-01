import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FileText, Download, Eye, BarChart3, Settings, TrendingUp, AlertCircle } from 'lucide-react';

const reports = [
  {
    id: '1',
    title: 'Model Training Report v2.3',
    type: 'Training Analysis',
    date: '2024-07-30',
    size: '12.4 MB',
    status: 'completed',
    description: 'Comprehensive analysis of latest model training iteration with Falcon synthetic data'
  },
  {
    id: '2',
    title: 'Performance Benchmarks Q3',
    type: 'Performance Review',
    date: '2024-07-28',
    size: '8.7 MB',
    status: 'completed',
    description: 'Quarterly performance metrics and comparison with previous model versions'
  },
  {
    id: '3',
    title: 'Failure Analysis Deep Dive',
    type: 'Error Analysis',
    date: '2024-07-25',
    size: '15.2 MB',
    status: 'completed',
    description: 'Detailed investigation of common failure patterns and proposed solutions'
  },
  {
    id: '4',
    title: 'EVA Equipment Detection Study',
    type: 'Use Case Study',
    date: '2024-07-20',
    size: '6.8 MB',
    status: 'draft',
    description: 'Specialized analysis for Extra-Vehicular Activity equipment monitoring'
  },
];

const trainingMetrics = [
  { label: 'Training Epochs', value: '156', trend: 'stable' },
  { label: 'Final Loss', value: '0.0234', trend: 'down' },
  { label: 'Validation Accuracy', value: '94.7%', trend: 'up' },
  { label: 'Learning Rate', value: '0.001', trend: 'stable' },
];

export default function ReportsViewer() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

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
            <FileText className="w-6 h-6 sm:w-8 sm:h-8 text-primary mb-2 sm:mb-0 sm:mr-3" />
            <h2 className="text-3xl sm:text-4xl font-bold">Training Reports</h2>
          </div>
          <p className="text-lg sm:text-xl text-muted-foreground">
            Comprehensive analysis and documentation of model development
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6 sm:gap-8 px-4">
          {/* Reports List */}
          <motion.div
            className="space-y-3 sm:space-y-4"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Available Reports</h3>
            
            {reports.map((report, index) => (
              <motion.div
                key={report.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card 
                  className={`glass-panel hover-tilt hover:cosmic-glow cursor-pointer transition-all duration-300 ${
                    selectedReport === report.id ? 'ring-2 ring-primary' : ''
                  }`}
                  onClick={() => setSelectedReport(report.id)}
                >
                  <div className="space-y-2 sm:space-y-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-xs sm:text-sm leading-tight truncate">{report.title}</h4>
                        <p className="text-xs text-muted-foreground mt-1">{report.type}</p>
                      </div>
                      <Badge 
                        variant={report.status === 'completed' ? 'default' : 'secondary'}
                        className="text-xs flex-shrink-0 ml-2"
                      >
                        {report.status}
                      </Badge>
                    </div>
                    
                    <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                      {report.description}
                    </p>
                    
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs text-muted-foreground">
                      <span>{report.date}</span>
                      <span>{report.size}</span>
                    </div>
                    
                    <div className="flex space-x-2 pt-2">
                      <Button size="sm" variant="outline" className="flex-1 text-xs">
                        <Eye className="w-3 h-3 mr-1" />
                        View
                      </Button>
                      <Button size="sm" variant="ghost" className="px-2">
                        <Download className="w-3 h-3" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </motion.div>

          {/* Report Viewer */}
          <motion.div
            className="lg:col-span-2 mt-6 lg:mt-0"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <Card className="glass-panel hover-tilt h-full min-h-[500px] sm:min-h-[600px]">
              {selectedReport ? (
                <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4 sm:space-y-6">
                  <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4 gap-1">
                    <TabsTrigger value="overview" className="text-xs sm:text-sm">Overview</TabsTrigger>
                    <TabsTrigger value="architecture" className="text-xs sm:text-sm">Architecture</TabsTrigger>
                    <TabsTrigger value="results" className="text-xs sm:text-sm">Results</TabsTrigger>
                    <TabsTrigger value="challenges" className="text-xs sm:text-sm">Challenges</TabsTrigger>
                  </TabsList>

                  <TabsContent value="overview" className="space-y-4 sm:space-y-6">
                    <div className="space-y-3 sm:space-y-4">
                      <h3 className="text-lg sm:text-xl font-semibold">Model Training Report v2.3</h3>
                      
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                        {trainingMetrics.map((metric) => (
                          <div key={metric.label} className="glass rounded-lg p-3 hover:cosmic-glow transition-all duration-300">
                            <div className="flex items-center justify-between">
                              <span className="text-xs sm:text-sm text-muted-foreground">{metric.label}</span>
                              <div className="flex items-center space-x-1">
                                {metric.trend === 'up' && <TrendingUp className="w-3 h-3 text-primary" />}
                                {metric.trend === 'down' && <TrendingUp className="w-3 h-3 text-accent rotate-180" />}
                                {metric.trend === 'stable' && <div className="w-3 h-3" />}
                              </div>
                            </div>
                            <div className="text-base sm:text-lg font-bold text-primary">{metric.value}</div>
                          </div>
                        ))}
                      </div>

                      <div className="space-y-3">
                        <h4 className="text-sm sm:text-base font-medium">Executive Summary</h4>
                        <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                          This report presents the results of our latest model training iteration using Falcon's 
                          synthetic digital twin data. The model achieved significant improvements in accuracy 
                          and precision, particularly in low-light and partially occluded scenarios.
                        </p>
                        
                        <div className="glass rounded-lg p-3 sm:p-4 bg-primary/5">
                          <h5 className="text-sm sm:text-base font-medium text-primary mb-2">Key Achievements</h5>
                          <ul className="text-xs sm:text-sm text-muted-foreground space-y-1">
                            <li>• mAP@0.5 improved from 91.2% to 94.7%</li>
                            <li>• 47% reduction in false positive rate</li>
                            <li>• Enhanced performance in challenging lighting conditions</li>
                            <li>• Improved detection of partially occluded objects</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="architecture" className="space-y-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold">Model Architecture</h3>
                      
                      {/* Architecture Diagram Placeholder */}
                      <div className="aspect-video bg-gradient-to-br from-secondary/20 to-card/50 rounded-lg p-6 flex items-center justify-center">
                        <div className="text-center space-y-2">
                          <BarChart3 className="w-12 h-12 text-primary mx-auto" />
                          <p className="text-sm text-muted-foreground">Model Architecture Visualization</p>
                          <p className="text-xs text-muted-foreground">YOLOv8-based detection network with custom space-domain adaptations</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="glass rounded-lg p-4">
                          <h4 className="font-medium mb-2">Network Details</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Base Model:</span>
                              <span>YOLOv8x</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Input Size:</span>
                              <span>640x640</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Parameters:</span>
                              <span>68.2M</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">FLOPs:</span>
                              <span>257.8G</span>
                            </div>
                          </div>
                        </div>

                        <div className="glass rounded-lg p-4">
                          <h4 className="font-medium mb-2">Training Setup</h4>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Batch Size:</span>
                              <span>16</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Optimizer:</span>
                              <span>AdamW</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Initial LR:</span>
                              <span>0.01</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Hardware:</span>
                              <span>4x RTX 4090</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="results" className="space-y-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold">Training Results</h3>
                      
                      {/* Training Chart Placeholder */}
                      <div className="aspect-video bg-gradient-to-br from-secondary/20 to-card/50 rounded-lg p-6">
                        <h4 className="font-medium mb-4">Training Loss Progression</h4>
                        <div className="h-full flex items-end justify-center space-x-2">
                          {Array.from({ length: 10 }, (_, i) => (
                            <motion.div
                              key={i}
                              className="bg-primary rounded-t w-8"
                              style={{ height: `${60 + Math.random() * 40}%` }}
                              initial={{ height: 0 }}
                              animate={{ height: `${60 + Math.random() * 40}%` }}
                              transition={{ duration: 1, delay: i * 0.1 }}
                            />
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
                        <div className="glass rounded-lg p-3 sm:p-4 text-center hover:cosmic-glow transition-all duration-300">
                          <div className="text-xl sm:text-2xl font-bold text-primary">94.7%</div>
                          <div className="text-xs sm:text-sm text-muted-foreground">mAP@0.5</div>
                        </div>
                        <div className="glass rounded-lg p-3 sm:p-4 text-center hover:aurora-glow transition-all duration-300">
                          <div className="text-xl sm:text-2xl font-bold text-accent">99.2%</div>
                          <div className="text-xs sm:text-sm text-muted-foreground">Precision</div>
                        </div>
                        <div className="glass rounded-lg p-3 sm:p-4 text-center hover:cosmic-glow transition-all duration-300">
                          <div className="text-xl sm:text-2xl font-bold text-primary">91.4%</div>
                          <div className="text-xs sm:text-sm text-muted-foreground">Recall</div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="challenges" className="space-y-6">
                    <div className="space-y-4">
                      <h3 className="text-xl font-semibold">Challenges & Solutions</h3>
                      
                      <div className="space-y-4">
                        {[
                          {
                            challenge: 'Low-light Detection',
                            description: 'Objects poorly visible in shadowed areas of space station',
                            solution: 'Enhanced data augmentation with synthetic low-light scenarios'
                          },
                          {
                            challenge: 'Partial Occlusion',
                            description: 'Equipment hidden behind structural elements or other objects',
                            solution: 'Multi-angle synthetic data generation using Falcon digital twin'
                          },
                          {
                            challenge: 'Scale Variation',
                            description: 'Objects appear at different scales depending on camera distance',
                            solution: 'Multi-scale training with varied camera positioning data'
                          }
                        ].map((item, index) => (
                          <motion.div
                            key={index}
                            className="glass rounded-lg p-4"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, delay: index * 0.1 }}
                            viewport={{ once: true }}
                          >
                            <div className="flex items-start space-x-3">
                              <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
                              <div className="flex-1">
                                <h4 className="font-medium text-destructive">{item.challenge}</h4>
                                <p className="text-sm text-muted-foreground mt-1">{item.description}</p>
                                <div className="mt-2 p-2 glass rounded bg-primary/5">
                                  <p className="text-sm"><strong>Solution:</strong> {item.solution}</p>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="h-64 sm:h-96 flex items-center justify-center text-center">
                  <div className="space-y-3">
                    <FileText className="w-8 h-8 sm:w-12 sm:h-12 text-muted-foreground mx-auto" />
                    <h3 className="text-base sm:text-lg font-medium">Select a Report</h3>
                    <p className="text-xs sm:text-sm text-muted-foreground px-4">
                      Choose a report from the list to view its contents
                    </p>
                  </div>
                </div>
              )}
            </Card>
          </motion.div>
        </div>
      </div>
    </section>
  );
}