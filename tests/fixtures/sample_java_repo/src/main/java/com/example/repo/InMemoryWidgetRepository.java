package com.example.repo;

import com.example.model.Widget;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * A basic {@link WidgetRepository} backed by an in-memory list.
 */
public class InMemoryWidgetRepository implements WidgetRepository {
    private final List<Widget> widgets = new ArrayList<>();

    @Override
    public void save(Widget widget) {
        widgets.add(widget);
    }

    @Override
    public Optional<Widget> findByName(String name) {
        for (Widget w : widgets) {
            if (w.getName().equals(name)) {
                return Optional.of(w);
            }
        }
        return Optional.empty();
    }

    @Override
    public List<Widget> findAll() {
        return new ArrayList<>(widgets);
    }
}
