#!/usr/bin/env python3
"""
SHIELD AI - Server Startup Script
Launches the FastAPI server with configured settings
"""

import os
import sys
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def setup_logging(debug: bool = False):
    """Configure logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic', 'numpy', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Start SHIELD AI Server')
    parser.add_argument(
        '--host',
        type=str,
        default=os.getenv('API_HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('API_PORT', '8000')),
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        default=os.getenv('DEBUG', 'false').lower() == 'true',
        help='Enable auto-reload (development only)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=int(os.getenv('WORKERS', '1')),
        help='Number of worker processes (default: 1)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=os.getenv('DEBUG', 'false').lower() == 'true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Check dependencies
    check_dependencies()
    
    logger.info("=" * 60)
    logger.info("SHIELD AI Safety API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Workers: {args.workers}")
    logger.info("=" * 60)
    
    try:
        import uvicorn
        
        # Run server
        if args.reload:
            # Development mode with reload
            uvicorn.run(
                "src.api.fastapi_server:app",
                host=args.host,
                port=args.port,
                reload=True,
                reload_dirs=[project_root],
                log_level="debug" if args.debug else "info"
            )
        else:
            # Production mode
            uvicorn.run(
                "src.api.fastapi_server:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level="info",
                access_log=True
            )
            
    except KeyboardInterrupt:
        logger.info("\nServer shutdown requested")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
