import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Smartphone, AlertTriangle, CheckCircle, MapPin, Clock, Zap } from 'lucide-react';

interface Alert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  location: string;
  timestamp: string;
  status: 'active' | 'acknowledged' | 'resolved';
}

const mockAlerts: Alert[] = [
  {
    id: '1',
    type: 'critical',
    title: 'Fire Extinguisher Missing',
    message: 'EVA safety equipment not detected in designated location',
    location: 'Module B - Section 3',
    timestamp: '2024-07-30T15:42:33Z',
    status: 'active'
  },
  {
    id: '2',
    type: 'warning',
    title: 'Toolbox Misplaced',
    message: 'Essential maintenance tools detected in unauthorized area',
    location: 'Module A - Docking Port',
    timestamp: '2024-07-30T14:28:17Z',
    status: 'acknowledged'
  },
  {
    id: '3',
    type: 'info',
    title: 'Oxygen Tank Secured',
    message: 'Backup oxygen supply properly stowed after EVA',
    location: 'Module C - Storage Bay',
    timestamp: '2024-07-30T13:15:45Z',
    status: 'resolved'
  },
];

export default function MobilePreview() {
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [currentView, setCurrentView] = useState<'home' | 'alerts' | 'scan'>('home');

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical': return 'text-destructive';
      case 'warning': return 'text-yellow-500';
      case 'info': return 'text-primary';
      default: return 'text-muted-foreground';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'destructive';
      case 'acknowledged': return 'secondary';
      case 'resolved': return 'default';
      default: return 'outline';
    }
  };

  return (
    <section className="py-20 px-4 relative">
      <div className="container mx-auto">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <div className="flex items-center justify-center mb-4">
            <Smartphone className="w-8 h-8 text-primary mr-3" />
            <h2 className="text-4xl font-bold">Astronaut Assistant</h2>
          </div>
          <p className="text-xl text-muted-foreground">
            Mobile interface for real-time space station monitoring
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Mobile Mockup */}
          <motion.div
            className="flex justify-center"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
          >
            {/* Phone Frame */}
            <div className="relative">
              <div className="w-80 h-[640px] bg-gradient-to-b from-gray-800 to-gray-900 rounded-[3rem] p-2 shadow-2xl">
                {/* Screen */}
                <div className="w-full h-full bg-background rounded-[2.5rem] overflow-hidden relative">
                  {/* Notch */}
                  <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-32 h-6 bg-gray-900 rounded-b-xl z-10" />
                  
                  {/* Status Bar */}
                  <div className="flex justify-between items-center px-6 pt-8 pb-2 text-xs text-muted-foreground">
                    <span>ISS-COM</span>
                    <div className="flex items-center space-x-1">
                      <span>15:42</span>
                      <div className="w-4 h-2 border border-primary rounded-sm">
                        <div className="w-3/4 h-full bg-primary rounded-sm" />
                      </div>
                    </div>
                  </div>

                  {/* App Content */}
                  <div className="px-4 pb-4 h-full">
                    <AnimatePresence mode="wait">
                      {currentView === 'home' && (
                        <motion.div
                          key="home"
                          className="space-y-4"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* Header */}
                          <div className="text-center py-4">
                            <h3 className="text-lg font-semibold">Orbital Sight</h3>
                            <p className="text-xs text-muted-foreground">Station Monitor Active</p>
                          </div>

                          {/* Quick Stats */}
                          <div className="grid grid-cols-2 gap-3">
                            <div className="glass rounded-lg p-3 text-center">
                              <div className="text-lg font-bold text-primary">23</div>
                              <div className="text-xs text-muted-foreground">Objects Tracked</div>
                            </div>
                            <div className="glass rounded-lg p-3 text-center">
                              <div className="text-lg font-bold text-accent">1</div>
                              <div className="text-xs text-muted-foreground">Active Alert</div>
                            </div>
                          </div>

                          {/* Critical Alert */}
                          <motion.div
                            className="glass rounded-lg p-3 border-l-4 border-destructive"
                            animate={{
                              boxShadow: [
                                '0 0 0 rgba(239, 68, 68, 0)',
                                '0 0 20px rgba(239, 68, 68, 0.3)',
                                '0 0 0 rgba(239, 68, 68, 0)',
                              ],
                            }}
                            transition={{
                              duration: 2,
                              repeat: Infinity,
                              ease: "easeInOut",
                            }}
                          >
                            <div className="flex items-start space-x-3">
                              <AlertTriangle className="w-5 h-5 text-destructive mt-0.5" />
                              <div className="flex-1 min-w-0">
                                <h4 className="font-medium text-sm">Fire Extinguisher Missing</h4>
                                <p className="text-xs text-muted-foreground mt-1">
                                  Module B - Section 3
                                </p>
                                <p className="text-xs text-destructive mt-2">
                                  Immediate attention required
                                </p>
                              </div>
                            </div>
                          </motion.div>

                          {/* Action Buttons */}
                          <div className="space-y-2 pt-4">
                            <Button 
                              size="sm" 
                              className="w-full" 
                              onClick={() => setCurrentView('alerts')}
                            >
                              <AlertTriangle className="w-4 h-4 mr-2" />
                              View All Alerts
                            </Button>
                            <Button 
                              variant="outline" 
                              size="sm" 
                              className="w-full"
                              onClick={() => setCurrentView('scan')}
                            >
                              <Zap className="w-4 h-4 mr-2" />
                              Start Manual Scan
                            </Button>
                          </div>
                        </motion.div>
                      )}

                      {currentView === 'alerts' && (
                        <motion.div
                          key="alerts"
                          className="space-y-3"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* Header */}
                          <div className="flex items-center justify-between py-2">
                            <h3 className="font-semibold">Active Alerts</h3>
                            <Button variant="ghost" size="sm" onClick={() => setCurrentView('home')}>
                              ←
                            </Button>
                          </div>

                          {/* Alerts List */}
                          <div className="space-y-2 max-h-96 overflow-y-auto">
                            {mockAlerts.map((alert) => (
                              <motion.div
                                key={alert.id}
                                className="glass rounded-lg p-3 cursor-pointer hover:bg-accent/5"
                                whileTap={{ scale: 0.98 }}
                                onClick={() => setSelectedAlert(alert)}
                              >
                                <div className="flex items-start space-x-3">
                                  <div className={`w-2 h-2 rounded-full mt-2 ${getAlertColor(alert.type).replace('text-', 'bg-')}`} />
                                  <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between">
                                      <h4 className="font-medium text-sm truncate">{alert.title}</h4>
                                      <Badge variant={getStatusColor(alert.status)} className="text-xs">
                                        {alert.status}
                                      </Badge>
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-1">{alert.location}</p>
                                    <p className="text-xs text-muted-foreground">
                                      {new Date(alert.timestamp).toLocaleTimeString()}
                                    </p>
                                  </div>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </motion.div>
                      )}

                      {currentView === 'scan' && (
                        <motion.div
                          key="scan"
                          className="space-y-4"
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: -20 }}
                          transition={{ duration: 0.3 }}
                        >
                          {/* Header */}
                          <div className="flex items-center justify-between py-2">
                            <h3 className="font-semibold">Manual Scan</h3>
                            <Button variant="ghost" size="sm" onClick={() => setCurrentView('home')}>
                              ←
                            </Button>
                          </div>

                          {/* Camera View */}
                          <div className="aspect-video bg-gradient-to-br from-secondary/30 to-card/50 rounded-lg relative overflow-hidden">
                            <div className="absolute inset-4 border-2 border-primary/30 rounded-lg" />
                            <div className="absolute top-2 left-2 glass rounded px-2 py-1 text-xs">
                              Live Feed
                            </div>
                            
                            {/* Scanning overlay */}
                            <motion.div
                              className="absolute inset-0 bg-gradient-to-b from-transparent via-primary/10 to-transparent"
                              animate={{ y: ['-100%', '100%'] }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            />
                          </div>

                          {/* Scan Results */}
                          <div className="space-y-2">
                            <h4 className="font-medium text-sm">Detected Objects</h4>
                            <div className="space-y-1">
                              {['Fire Extinguisher', 'Toolbox', 'Oxygen Tank'].map((item, index) => (
                                <div key={item} className="flex items-center justify-between glass rounded p-2">
                                  <span className="text-sm">{item}</span>
                                  <CheckCircle className="w-4 h-4 text-primary" />
                                </div>
                              ))}
                            </div>
                          </div>

                          <Button size="sm" className="w-full">
                            Complete Scan
                          </Button>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Features Description */}
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <div className="space-y-6">
              <h3 className="text-2xl font-bold">Real-Time Monitoring</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Astronauts receive instant alerts about missing or misplaced equipment, 
                ensuring mission safety and operational efficiency in zero gravity environments.
              </p>
            </div>

            <div className="space-y-4">
              {[
                {
                  icon: AlertTriangle,
                  title: 'Critical Alerts',
                  description: 'Immediate notifications for safety-critical equipment status'
                },
                {
                  icon: MapPin,
                  title: 'Location Tracking',
                  description: 'Precise object positioning within station modules'
                },
                {
                  icon: Clock,
                  title: 'Timeline History',
                  description: 'Complete audit trail of object movements and status changes'
                },
                {
                  icon: Zap,
                  title: 'Manual Override',
                  description: 'On-demand scanning capability for immediate verification'
                }
              ].map((feature, index) => (
                <motion.div
                  key={feature.title}
                  className="flex items-start space-x-4"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.6 + index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <div className="p-2 rounded-lg bg-gradient-cosmic text-primary">
                    <feature.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="font-semibold">{feature.title}</h4>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="glass-card bg-gradient-cosmic/10">
              <h4 className="font-semibold mb-2">Use Case Example</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                During EVA preparation, the system detects that a fire extinguisher is missing from its 
                designated location in Module B. An immediate alert is sent to all crew members' devices, 
                preventing potential safety hazards and ensuring proper equipment accountability.
              </p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}