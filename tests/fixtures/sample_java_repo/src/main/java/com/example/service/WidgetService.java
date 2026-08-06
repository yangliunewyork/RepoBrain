package com.example.service;

import com.example.model.Widget;
import com.example.repo.WidgetRepository;
import java.util.List;
import java.util.Optional;

/**
 * Application service coordinating widget lookups and creation.
 */
public class WidgetService {
    private final WidgetRepository repository;

    public WidgetService(WidgetRepository repository) {
        this.repository = repository;
    }

    /**
     * Creates and stores a new widget.
     *
     * @param name  the widget name
     * @param price the widget price
     * @return the created widget
     */
    public Widget createWidget(String name, double price) {
        Widget widget = new Widget(name, price);
        repository.save(widget);
        return widget;
    }

    public Optional<Widget> findWidget(String name) {
        return repository.findByName(name);
    }

    public List<Widget> listWidgets() {
        return repository.findAll();
    }
}
