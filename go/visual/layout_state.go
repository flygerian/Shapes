package visual

type frame struct {
	Point
	Area
}

type layoutAware interface {
	move(point Point)
	resize(area Area)
	layout() frame
}

type layoutState struct {
	point Point
	area  Area
}

func (l *layoutState) move(point Point) {
	l.point = point
}

func (l *layoutState) resize(area Area) {
	l.area = area
}

func (l *layoutState) layout() frame {
	return frame{
		Point: l.point,
		Area:  l.area,
	}
}

func setLayout(artefact Artefact, point Point, area Area) {
	if artefact == nil {
		return
	}
	node, ok := artefact.(layoutAware)
	if !ok {
		return
	}
	node.move(point)
	node.resize(area)
}

func layoutOf(artefact Artefact) frame {
	if artefact == nil {
		return frame{}
	}
	node, ok := artefact.(layoutAware)
	if !ok {
		return frame{}
	}
	return node.layout()
}
