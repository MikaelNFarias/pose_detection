import 'dart:math';  // Adiciona esta importação para usar funções matemáticas
import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:sensors_plus/sensors_plus.dart';

import 'detector_view.dart';
import 'painters/pose_painter.dart';

class PoseDetectorView extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _PoseDetectorViewState();
}

class _PoseDetectorViewState extends State<PoseDetectorView> {
  final PoseDetector _poseDetector =
      PoseDetector(options: PoseDetectorOptions());
  bool _canProcess = true;
  bool _isBusy = false;
  CustomPaint? _customPaint;
  String? _text;
  var _cameraLensDirection = CameraLensDirection.back;
  AccelerometerEvent? _accelerometerEvent;
  DateTime? _lastAccelerometerUpdate;
  int? _lastInterval;

  @override
  void initState() {
    super.initState();
    accelerometerEvents.listen((event) {
      final now = DateTime.now();
      if (_lastAccelerometerUpdate != null) {
        final interval = now.difference(_lastAccelerometerUpdate!);
        setState(() {
          _lastInterval = interval.inMilliseconds;
        });
      }
      _lastAccelerometerUpdate = now;
      setState(() {
        _accelerometerEvent = event;
      });
    });
  }

  @override
  void dispose() async {
    _canProcess = false;
    _poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Pose Detector'),
      ),
      body: Stack(
        children: [
          DetectorView(
            title: 'Pose Detector',
            customPaint: _customPaint,
            text: _text,
            onImage: _processImage,
            initialCameraLensDirection: _cameraLensDirection,
            onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
          ),
          Positioned(
            bottom: 80,  // Ajuste para deixar espaço acima da barra deslizante
            left: 16,
            child: _buildAccelerometerData(),
          ),
        ],
      ),
    );
  }

  Widget _buildAccelerometerData() {
    if (_accelerometerEvent == null) {
      return Text(
        'Acelerômetro: Carregando...',
        style: TextStyle(fontSize: 16, color: Colors.white),
      );
    }

    final x = _accelerometerEvent!.x.toStringAsFixed(2);
    final y = _accelerometerEvent!.y.toStringAsFixed(2);
    final z = _accelerometerEvent!.z.toStringAsFixed(2);
    final inclinationX = _calculateInclinationX().toStringAsFixed(2);
    final inclinationY = _calculateInclinationY().toStringAsFixed(2);
    final interval = _lastInterval != null ? '${_lastInterval} ms' : 'N/A';

    return Text(
      'Acelerômetro:\nX: $x, Y: $y, Z: $z\n'
      'Inclinação X (Retrato): $inclinationX°\n'
      'Inclinação Y (Paisagem): $inclinationY°\n'
      'Intervalo: $interval',
      style: TextStyle(fontSize: 16, color: Colors.white),
    );
  }

  double _calculateInclinationX() {
    if (_accelerometerEvent == null) return 0.0;
    final angle = atan2(_accelerometerEvent!.y, _accelerometerEvent!.z);
    return angle * 180 / pi;
  }

  double _calculateInclinationY() {
    if (_accelerometerEvent == null) return 0.0;
    final angle = atan2(_accelerometerEvent!.x, _accelerometerEvent!.z);
    return angle * 180 / pi;
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_canProcess) return;
    if (_isBusy) return;
    _isBusy = true;
    setState(() {
      _text = '';
    });
    final poses = await _poseDetector.processImage(inputImage);
    if (inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      final painter = PosePainter(
        poses,
        inputImage.metadata!.size,
        inputImage.metadata!.rotation,
        _cameraLensDirection,
      );
      _customPaint = CustomPaint(painter: painter);
    } else {
      _text = 'Poses found: ${poses.length}\n\n';
      _customPaint = null;
    }
    _isBusy = false;
    if (mounted) {
      setState(() {});
    }
  }
}
