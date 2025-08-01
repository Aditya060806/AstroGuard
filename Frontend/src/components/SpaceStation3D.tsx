import { motion } from 'framer-motion';

export default function SpaceStation3D() {
  return (
    <div className="w-full h-64 relative flex items-center justify-center">
      {/* CSS-based 3D space station animation */}
      <motion.div
        className="relative"
        animate={{ rotateY: 360 }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        style={{ transformStyle: "preserve-3d" }}
      >
        {/* Main station core */}
        <motion.div
          className="w-16 h-24 bg-gradient-to-b from-primary to-primary/70 rounded-lg relative mx-auto"
          animate={{ rotateX: [0, 10, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        >
          {/* Solar panels */}
          <div className="absolute top-1/2 -left-12 w-8 h-16 bg-gradient-to-r from-blue-500 to-blue-400 rounded transform -translate-y-1/2 opacity-80" />
          <div className="absolute top-1/2 -right-12 w-8 h-16 bg-gradient-to-r from-blue-500 to-blue-400 rounded transform -translate-y-1/2 opacity-80" />
          
          {/* Communication dishes */}
          <div className="absolute -top-2 left-1/2 w-4 h-4 bg-gradient-to-br from-accent to-accent/70 rounded-full transform -translate-x-1/2" />
          <div className="absolute -bottom-2 left-1/2 w-3 h-3 bg-gradient-to-br from-green-400 to-green-300 rounded-full transform -translate-x-1/2" />
          
          {/* Docking ports */}
          <div className="absolute top-1/2 -front-2 w-2 h-3 bg-gradient-to-b from-orange-400 to-orange-300 rounded transform -translate-y-1/2" />
          <div className="absolute top-1/2 -back-2 w-2 h-3 bg-gradient-to-b from-orange-400 to-orange-300 rounded transform -translate-y-1/2" />
        </motion.div>
      </motion.div>

      {/* Floating particles */}
      {Array.from({ length: 8 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-primary/60 rounded-full"
          style={{
            left: `${20 + Math.random() * 60}%`,
            top: `${20 + Math.random() * 60}%`,
          }}
          animate={{
            scale: [0, 1, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: 3 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  );
}